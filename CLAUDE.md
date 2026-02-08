# StarSpace — Claude Code Session Guide

Single source of truth for working on this repository.

## Environment Setup

```bash
pip install numpy numba
make BOOST_DIR=/usr/include opt    # native C++ binary (for comparison)
```

## Project Structure

```
starspace.py              — Pure Python StarSpace (numpy + numba), all 6 modes
test_starspace_modes.py   — Test suite (modes 0-5, file I/O, save/load)
starspace.md              — Algorithm notes
src/                      — Original C++ StarSpace source
makefile                  — C++ build (use BOOST_DIR=/usr/include)
```

## Running Tests

```bash
python3 test_starspace_modes.py
```

All 8 tests must pass (modes 0-5, file input, iter_lines input).

## Architecture: starspace.py

### Pipeline (retokenise-per-epoch, no intermediate arrays)

Training operates directly on a memory-mapped text file:

1. **Pass 1** `_vocab_scan` (Numba `@njit`): Scans mmap'd `uint8` bytes → open-addressing hash table. FNV-1a hashing, `__label__` prefix detection, frequency counting. Returns status for hash table overflow handling.

2. **Pass 2** `_extract_vocab` (Python): Filters words by `min_count`, sorts by frequency descending, assigns final embedding IDs, builds remap array. Reconstructs Python strings from byte offsets for the `Vocab` dataclass.

3. **Train** `_train_epoch` (Numba `@njit`): Retokenises from mmap each epoch — scans bytes, looks up tokens in hash table via `_lookup_token`, builds LHS context via `_build_word_ctx`, trains inline via `_train_step`. No intermediate arrays stored between passes.

### Key functions

| Function | Type | Purpose |
|---|---|---|
| `_fnv1a_bytes(data)` | `@njit` | FNV-1a 32-bit hash (matches C++ StarSpace/fastText) |
| `_vocab_scan(buf, ...)` | `@njit` | Pass 1: mmap bytes → hash table |
| `_lookup_token(buf, ...)` | `@njit` | Hash table lookup for one token → (id, fnv) |
| `_build_word_ctx(wids, ...)` | `@njit` | Build LHS context: word IDs + n-gram buckets |
| `_train_step(emb, ...)` | `@njit` | One hinge-loss + neg-sampling + AdaGrad step |
| `_train_epoch(emb, ...)` | `@njit` | Scan mmap, tokenise, mode dispatch, call helpers |
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
| 5 | context words | target word |

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
adagrad          28 MB   n_emb × float32
epoch scratch  ~320 KB   line buffers, ctx_buf, vectors (negligible)
─────────────────────
total         ~3.5 GB    the text is never copied into RAM
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

### Constraints

- Only numpy and numba as dependencies
- All 6 training modes must work
- File-based training preferred (iterables spill to temp file)
- All `@njit` helpers must be top-level functions (Numba doesn't support closures)
