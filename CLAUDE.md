# StarSpace — Claude Code Session Guide

Single source of truth for working on this repository.

## Environment Setup

```bash
pip install numpy numba
# macOS: make BOOST_DIR=/opt/homebrew/include opt
# Linux: make BOOST_DIR=/usr/include opt
make BOOST_DIR=/opt/homebrew/include opt    # native C++ binary (for comparison)
```

## Project Structure

```
starspace.py              — Pure Python implementation (numpy + numba)
test_starspace.py         — Self-consistency tests (determinism, convergence, API)
test_starspace_native.py  — Native C++ comparison tests (all modes, parametrized)
test_starspace_lastfm.py  — Last.FM tests (artist recommendation + tag co-occurrence)
test_starspace_papers.py  — Paper recommendation (collaborative filtering + content)
data/lastfm/              — Last.FM hetrec2011-2k dataset
src/                      — Original C++ source
```

## Running Tests

```bash
# All tests
python3 -m pytest test_starspace.py test_starspace_native.py -q
```

**test_starspace.py** tests determinism, save/load roundtrip, convergence for all 5 modes, and API surface (predict, file input, iterable input).

**test_starspace_native.py** compares vocabulary, embedding norms, P@1, and semantic predictions (both Python and native) against the native C++ binary on real data (tagged_post.txt). Parametrized over all 5 modes.

**test_starspace_papers.py** trains mode 1 on paper co-occurrence data from `~/work/data/papers/`. Three co-occurrence sources: (user, collection) groups, project source groups, and user-level library groups. Single-paper lines for all 223K papers ensure full library coverage. Features: title words, AU_-prefixed author names, J_-prefixed journal tokens. Evaluation: collaborative filtering holdout, content-based retrieval, project leave-one-out, and per-source-voting novel recommendations.

## Code Quality

### Python linting

**Configuration**: `.flake8` in project root (globally ignores E501, E221, F841)

```bash
# Check code style (uses .flake8 config automatically)
python3 -m flake8 starspace.py test_*.py

# Check static types
pyright starspace.py test_*.py
```

**Style:**

- PEP 8 compliant (E501, E221, F841 ignored globally)
- 2 blank lines between top-level definitions
- Import order: stdlib → third-party → local
- Pyright must pass with 0 errors

## Architecture: starspace.py

### Pipeline (retokenise-per-epoch, no intermediate arrays)

Training operates directly on a memory-mapped text file:

1. **Pass 1** `_vocab_scan` (Numba `@njit`): Scans mmap'd `uint8` bytes → open-addressing hash table. FNV-1a hashing, `__label__` prefix detection, frequency counting. Returns status for hash table overflow handling.

2. **Pass 2** `_extract_vocab` (Python): Filters words by `min_count`, sorts by frequency descending, assigns final embedding IDs, builds remap array. Reconstructs Python strings from byte offsets for the `Vocab` dataclass.

3. **Train** `_train_epoch` (Numba `@njit`): Retokenises from mmap each epoch — scans bytes, looks up tokens in hash table via `_lookup_token`, builds LHS context via `_build_word_ctx`, buffers examples into batches, trains via `_train_batch` (shared negatives across batch, matching native `trainOneBatch`). No intermediate arrays stored between passes.

### Key functions

| Function | Type | Purpose |
|---|---|---|
| `_fnv1a_bytes(data)` | `@njit` | FNV-1a 32-bit hash (matches C++ StarSpace/fastText) |
| `_vocab_scan(buf, ...)` | `@njit` | Pass 1: mmap bytes → hash table |
| `_lookup_token(buf, ...)` | `@njit` | Hash table lookup for one token → (id, fnv) |
| `_build_word_ctx(wids, ...)` | `@njit` | Build LHS context: word IDs + n-gram buckets |
| `_train_batch(emb, ...)` | `@njit` | Batch training: shared negatives across batch (matches native `trainOneBatch`) |
| `_train_epoch(emb, ...)` | `@njit` | Scan mmap, tokenise, mode dispatch, buffer batches, call helpers |
| `_alloc_hash_table(n)` | Python | Allocate struct-of-arrays hash table |
| `_extract_vocab(buf, ...)` | Python | Pass 2: hash table → Vocab + remap |

### Embedding matrix layout

```
[words (0..nwords) | labels (nwords..nwords+nlabels) | ngram buckets] × dim
```

### Training modes

| Mode | LHS | RHS |
|------|-----|-----|
| 0 | words | 1 random label |
| 1 | words + rest labels | 1 random label |
| 2 | words + 1 label | rest labels (multi-RHS) |
| 3 | words + 1 label | 1 other label |
| 4 | words + label[0] | label[1] |

### Hash table design

Struct-of-arrays with open addressing (linear probing):
- Power-of-2 size, mask-based slot computation
- 6 parallel arrays: `fnv`, `occupied`, `tok_start`, `tok_len`, `freq`, `is_label`
- Collision handling: FNV match + length match + byte-by-byte comparison
- Load factor threshold: 70% triggers overflow → Python doubles size and retries

### Memory footprint (10 GB text file, ~5 M unique words, dim=100)

Hash table is 33.5 M slots (power-of-2 ≥ 5M × 4). After Pass 2, `ht_freq` (268 MB)
and `ht_is_label` (33.5 MB) are deleted — only the 4 arrays needed by `_lookup_token`
during retokenisation are kept. No intermediate training arrays are stored.

```
text mmap          0 B   OS pages from disk, not RAM
hash table      570 MB   4 kept arrays × 33.5 M slots (fnv/occ/tok_start/tok_len)
remap           134 MB   table_size × int32
embeddings    2,800 MB   (5 M + labels + 2 M buckets) × 100 × float32
adagrad_lhs      28 MB   n_emb × float32 (separate LHS accumulator)
adagrad_rhs      28 MB   n_emb × float32 (separate RHS accumulator)
epoch scratch  ~320 KB   line buffers, ctx_buf, vectors (negligible)
─────────────────────
total         ~3.6 GB    the text is never copied into RAM
```

### API

```python
model = StarSpace.train("train.txt", dim=100, epoch=5)
model = StarSpace.train([["__label__pos", "great"]], dim=100)  # spills to temp file
model.predict("great movie", k=3)      # → [("__label__pos", 0.92), ...]
model.test("test.txt", k=1)            # → (N, precision@k, recall@k)
model.save("model.npz")
model = StarSpace.load("model.npz")
```

### Native C++ alignment

The Python implementation matches native C++ StarSpace behavior:
- **Batch training**: `batch_size=5` (default), shared negatives across batch matching native `trainOneBatch`
- **Separate AdaGrad**: LHS and RHS have independent accumulators (`adagrad_lhs`/`adagrad_rhs`), matching native `LHSUpdates_`/`RHSUpdates_`
- **Gradient rates**: LHS AdaGrad uses `||gradW||²/dim`, RHS positive rate = `lr`, RHS negative rate = `lr/num_violated_negs`
- **maxNegSamples cap**: Default 10, matching native `args_->maxNegSamples`
- **LR schedule**: Stepwise decay every 1000 samples within each epoch, epoch-level decay from `lr` to ~1e-9
- **Per-epoch line shuffling**: Fisher-Yates shuffle on line offsets
- **Negative sampling**: Pre-built pool matching native `getRandomRHS`. All label modes (0-4) weight by `1/|labels_in_example|` to match C++'s uniform-example-then-uniform-label sampling. Modes 1-4 also filter to multi-label examples only (matching native `parser.check()`). Mode 2 uses multi-label negatives (all labels from random example minus one)

Known differences from native:
- **PRNG implementation**: Native uses C `rand()` (unseeded) + `std::minstd_rand` for init; Python uses `np.random.RandomState(seed)` + custom LCG. Different sequences cause different shuffle orders, label selections, and negative samples. Norm ratios match within ~2%, but P@1 on small test sets varies by a few pp.
- **Vocab ordering**: Native uses unstable `std::sort`; tie-breaking for same-frequency words is undefined (word/label sets match, order may differ)

### Constraints

- Only numpy and numba as dependencies
- Supervised training modes 0-4 must work
- File-based training preferred (iterables spill to temp file)
- All `@njit` helpers must be top-level functions (Numba doesn't support closures)
