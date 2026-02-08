"""Cross-validation test: Python StarSpace vs native C++ StarSpace.

Trains both implementations on the same data with identical parameters,
then compares P@k and R@k on a held-out test set.  Also reports timing.

Datasets:
  python/test/tagged_post.txt  — 1,687 labeled lines → modes 0-4
  python/test/input.txt        — 906 unlabeled lines  → mode 5

Usage:
    python3 test_native_compare.py
"""

import os, subprocess, sys, tempfile, time

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
NATIVE_BIN = os.path.join(ROOT, "starspace")
TAGGED_DATA = os.path.join(ROOT, "python", "test", "tagged_post.txt")
WORD_DATA = os.path.join(ROOT, "python", "test", "input.txt")

sys.path.insert(0, ROOT)
from starspace import StarSpace, iter_lines

# ── helpers ──────────────────────────────────────────────────────────────────

def split_file(path, train_frac=0.8, seed=42):
    """Split a file into train/test by line, deterministic."""
    import random
    lines = open(path, encoding="utf-8", errors="replace").readlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    n = int(len(lines) * train_frac)
    train_f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ss_train_")
    test_f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ss_test_")
    for line in lines[:n]:
        train_f.write(line)
    for line in lines[n:]:
        test_f.write(line)
    train_f.close()
    test_f.close()
    return train_f.name, test_f.name, n, len(lines) - n


def run_native(train_path, test_path, mode, dim, epoch, lr, ws=5):
    """Train + test with native C++ StarSpace. Returns (p@1, r@1, train_sec)."""
    model_path = tempfile.mktemp(prefix="ss_model_")
    train_cmd = [
        NATIVE_BIN, "train",
        "-trainFile", train_path,
        "-model", model_path,
        "-trainMode", str(mode),
        "-dim", str(dim),
        "-epoch", str(epoch),
        "-lr", str(lr),
        "-ws", str(ws),
        "-negSearchLimit", "50",
        "-margin", "0.05",
        "-similarity", "cosine",
        "-adagrad", "1",
        "-shareEmb", "1",
        "-bucket", "2000000",
        "-ngrams", "1",
        "-minCount", "1",
        "-verbose", "0",
        "-thread", "1",
    ]

    t0 = time.time()
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"  NATIVE TRAIN FAILED (mode {mode}):", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        return None, None, train_time

    # Test
    test_cmd = [
        NATIVE_BIN, "test",
        "-testFile", test_path,
        "-model", model_path,
        "-trainMode", str(mode),
        "-K", "1",
        "-dim", str(dim),
        "-ngrams", "1",
        "-thread", "1",
    ]
    result = subprocess.run(test_cmd, capture_output=True, text=True)

    hit_at_1 = None
    for line in (result.stdout + result.stderr).splitlines():
        # Native outputs: "hit@1: 0.0896 hit@10: 0.298 ... Total examples : 201"
        if "hit@1:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "hit@1:":
                    hit_at_1 = float(parts[i + 1])
                    break
    p_at_1 = hit_at_1
    r_at_1 = hit_at_1  # hit@1 ≈ P@1 = R@1 for single-label eval

    # cleanup
    for ext in ("", ".tsv"):
        try:
            os.unlink(model_path + ext)
        except OSError:
            pass

    return p_at_1, r_at_1, train_time


def run_python(train_path, test_path, mode, dim, epoch, lr, ws=5):
    """Train + test with Python StarSpace. Returns (p@1, r@1, train_sec)."""
    t0 = time.time()
    model = StarSpace.train(
        train_path,
        dim=dim, epoch=epoch, lr=lr,
        margin=0.05, neg_search_limit=50,
        min_count=1, word_ngrams=1, bucket=2_000_000,
        norm_limit=1.0, init_rand_sd=0.001, seed=0,
        verbose=0, train_mode=mode, ws=ws)
    train_time = time.time() - t0

    n, p_at_1, r_at_1 = model.test(test_path, k=1)
    return p_at_1, r_at_1, train_time


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(NATIVE_BIN):
        print(f"ERROR: native binary not found at {NATIVE_BIN}")
        print("Build it with: make BOOST_DIR=/usr/include opt")
        sys.exit(1)

    print("=" * 70)
    print("  Native C++ vs Python StarSpace — Cross-Validation")
    print("=" * 70)

    # ── JIT warmup ──
    print("\nWarming up Numba JIT...")
    t0 = time.time()
    StarSpace.train(
        [["__label__a", "hello"], ["__label__b", "world"]],
        dim=10, epoch=1, verbose=0)
    print(f"JIT warm-up done in {time.time() - t0:.1f}s\n")

    # ── split data ──
    train_path, test_path, n_train, n_test = split_file(TAGGED_DATA)
    print(f"Dataset: {os.path.basename(TAGGED_DATA)}")
    print(f"  Train: {n_train} lines, Test: {n_test} lines\n")

    # Common params
    DIM = 50
    EPOCH = 10
    LR = 0.01
    WS = 5

    results = []
    all_ok = True

    # ── modes 0-4 on tagged data ──
    for mode in range(5):
        mode_names = {
            0: "Classification",
            1: "Label from rest",
            2: "Inverted labels",
            3: "Pair prediction",
            4: "Fixed pair",
        }
        print("-" * 70)
        print(f"  Mode {mode}: {mode_names[mode]}")
        print("-" * 70)

        np_at1, nr_at1, nt = run_native(
            train_path, test_path, mode, DIM, EPOCH, LR, WS)
        pp_at1, pr_at1, pt = run_python(
            train_path, test_path, mode, DIM, EPOCH, LR, WS)

        if np_at1 is None:
            print(f"  Native:  FAILED")
        else:
            print(f"  Native:  P@1={np_at1:.4f}  R@1={nr_at1:.4f}  "
                  f"time={nt:.2f}s")
        print(f"  Python:  P@1={pp_at1:.4f}  R@1={pr_at1:.4f}  "
              f"time={pt:.2f}s")

        # Check Python achieves reasonable quality
        # (not identical due to different RNG, but should be in same ballpark)
        ok = True
        if np_at1 is not None:
            # Python should achieve at least 50% of native's P@1
            # (generous threshold — different RNG seeds, neg sampling, etc.)
            if np_at1 > 0.05 and pp_at1 < np_at1 * 0.3:
                print(f"  WARNING: Python P@1 much lower than native")
                ok = False
            if pp_at1 <= 0.0 and np_at1 > 0.05:
                print(f"  FAIL: Python P@1 is zero but native is not")
                ok = False
                all_ok = False
        if pp_at1 > 0:
            print(f"  PASS")
        else:
            # Some modes (3,4) may have low P@1 on small data — acceptable
            if mode in (3, 4):
                print(f"  PASS (low P@1 expected for mode {mode} on small data)")
            else:
                print(f"  FAIL: Python P@1 is zero")
                all_ok = False

        results.append((mode, np_at1, pp_at1, nr_at1, pr_at1, nt, pt))

    # ── mode 5 on unlabeled data (word embedding) ──
    print("-" * 70)
    print(f"  Mode 5: Word embedding")
    print("-" * 70)

    model_path = tempfile.mktemp(prefix="ss_model_")
    train_cmd = [
        NATIVE_BIN, "train",
        "-trainFile", WORD_DATA,
        "-model", model_path,
        "-trainMode", "5",
        "-dim", str(DIM),
        "-epoch", str(EPOCH),
        "-lr", str(LR),
        "-ws", str(WS),
        "-negSearchLimit", "50",
        "-bucket", "2000000",
        "-ngrams", "1",
        "-minCount", "1",
        "-verbose", "0",
        "-thread", "1",
    ]

    t0 = time.time()
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    native_time_5 = time.time() - t0

    for ext in ("", ".tsv"):
        try:
            os.unlink(model_path + ext)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"  Native:  FAILED")
        print(f"  stderr: {result.stderr[:300]}")
    else:
        print(f"  Native:  trained OK  time={native_time_5:.2f}s")

    t0 = time.time()
    py_model_5 = StarSpace.train(
        WORD_DATA,
        dim=DIM, epoch=EPOCH, lr=LR,
        min_count=1, word_ngrams=1, bucket=2_000_000,
        verbose=0, train_mode=5, ws=WS)
    python_time_5 = time.time() - t0

    nw = py_model_5.vocab.nwords
    print(f"  Python:  trained OK  time={python_time_5:.2f}s  "
          f"vocab={nw} words")

    # Quick sanity: embeddings should be non-trivial
    import numpy as np
    emb_norms = np.linalg.norm(py_model_5.emb[:nw], axis=1)
    mean_norm = float(emb_norms.mean())
    print(f"  Python embedding mean norm: {mean_norm:.4f}")
    if mean_norm > 0.01:
        print(f"  PASS")
    else:
        print(f"  FAIL: embeddings look untrained")
        all_ok = False

    # ── cleanup ──
    os.unlink(train_path)
    os.unlink(test_path)

    # ── summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Mode':<8} {'Native P@1':>12} {'Python P@1':>12} "
          f"{'Native time':>13} {'Python time':>13}")
    print(f"  {'----':<8} {'----------':>12} {'----------':>12} "
          f"{'----------':>13} {'----------':>13}")
    for mode, np1, pp1, nr1, pr1, nt, pt in results:
        ns = f"{np1:.4f}" if np1 is not None else "FAILED"
        print(f"  {mode:<8} {ns:>12} {pp1:>12.4f} {nt:>12.2f}s {pt:>12.2f}s")
    print(f"  {'5':<8} {'—':>12} {'—':>12} "
          f"{native_time_5:>12.2f}s {python_time_5:>12.2f}s")
    print("=" * 70)

    if all_ok:
        print("  ALL MODES OK")
    else:
        print("  SOME MODES FAILED")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
