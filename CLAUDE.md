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

### Pipeline (no temp files)

Training operates directly on a memory-mapped text file:

1. **Pass 1** `_vocab_scan` (Numba `@njit`): Scans mmap'd `uint8` bytes → open-addressing hash table + line byte offsets. FNV-1a hashing, `__label__` prefix detection, frequency counting. Returns status for hash table overflow handling.

2. **Pass 2** `_extract_vocab` (Python): Filters words by `min_count`, sorts by frequency descending, assigns final embedding IDs, builds remap array. Reconstructs Python strings from byte offsets for the `Vocab` dataclass.

3. **Pass 3** `_build_training` (Numba `@njit`): Re-scans mmap'd bytes, tokenizes with hash table lookup + remap, applies mode-specific LHS/RHS logic, fills 8 in-memory arrays (ids, hashes, labels, extra, offsets × 3, neg_pool).

4. **Train** `_train_epoch` (Numba `@njit`): Hinge loss + cosine similarity + AdaGrad on the in-memory arrays. One call per epoch.

### Key functions

| Function | Type | Purpose |
|---|---|---|
| `_fnv1a_bytes(data)` | `@njit` | FNV-1a 32-bit hash (matches C++ StarSpace/fastText) |
| `_vocab_scan(buf, ...)` | `@njit` | Pass 1: mmap bytes → hash table + line offsets |
| `_build_training(buf, ...)` | `@njit` | Pass 3: mmap bytes → in-memory training arrays |
| `_train_epoch(emb, ...)` | `@njit` | Training kernel (hinge loss, AdaGrad) |
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
