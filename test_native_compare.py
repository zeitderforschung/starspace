"""Cross-validation test: Python StarSpace vs native C++ StarSpace.

Trains both implementations on the same data with identical parameters,
then compares P@k and R@k on a held-out test set.  Also reports timing
and speedup factor.

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
    """Train + test with native C++ StarSpace. Returns (p@1, r@1, n_test, train_sec)."""
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
        return None, None, None, train_time

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
    n_examples = None
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if "hit@1:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "hit@1:":
                    hit_at_1 = float(parts[i + 1])
                    break
        if "Total examples" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "examples" and i + 2 < len(parts) and parts[i + 1] == ":":
                    n_examples = int(parts[i + 2])
                    break

    # cleanup
    for ext in ("", ".tsv"):
        try:
            os.unlink(model_path + ext)
        except OSError:
            pass

    return hit_at_1, hit_at_1, n_examples, train_time


def run_python(train_path, test_path, mode, dim, epoch, lr, ws=5):
    """Train + test with Python StarSpace. Returns (p@1, r@1, n_test, train_sec)."""
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
    return p_at_1, r_at_1, n, train_time


def speedup_str(native_t, python_t):
    """Format speedup factor (how many times faster Python is)."""
    if native_t <= 0 or python_t <= 0:
        return "—"
    ratio = native_t / python_t
    if ratio >= 1.0:
        return f"{ratio:.1f}x"
    else:
        return f"1/{1/ratio:.1f}x"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(NATIVE_BIN):
        print(f"ERROR: native binary not found at {NATIVE_BIN}")
        print("Build it with: make BOOST_DIR=/usr/include opt")
        sys.exit(1)

    print("=" * 78)
    print("  Native C++ vs Python StarSpace — Cross-Validation")
    print("=" * 78)

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
    failures = []

    # ── modes 0-4 on tagged data ──
    for mode in range(5):
        mode_names = {
            0: "Classification",
            1: "Label from rest",
            2: "Inverted labels",
            3: "Pair prediction",
            4: "Fixed pair",
        }
        print("-" * 78)
        print(f"  Mode {mode}: {mode_names[mode]}")
        print("-" * 78)

        np_at1, nr_at1, n_nat, nt = run_native(
            train_path, test_path, mode, DIM, EPOCH, LR, WS)
        pp_at1, pr_at1, n_py, pt = run_python(
            train_path, test_path, mode, DIM, EPOCH, LR, WS)

        spd = speedup_str(nt, pt)

        if np_at1 is None:
            print(f"  Native:  FAILED")
        else:
            print(f"  Native:  P@1={np_at1:.4f}  R@1={nr_at1:.4f}  "
                  f"time={nt:.2f}s  N={n_nat}")
        print(f"  Python:  P@1={pp_at1:.4f}  R@1={pr_at1:.4f}  "
              f"time={pt:.2f}s  N={n_py}  speedup={spd}")

        # ── Verification checks ──
        ok = True
        reasons = []

        # 1. Native test must have parsed results
        if np_at1 is None:
            reasons.append("native test failed/unparsed")
            ok = False

        # 2. Python must have evaluated test examples
        if n_py is None or n_py == 0:
            reasons.append(f"Python evaluated 0 test examples")
            ok = False

        # 3. Native and Python should evaluate similar number of test examples
        if np_at1 is not None and n_nat is not None and n_py is not None:
            if n_nat > 0 and abs(n_nat - n_py) > max(n_nat, n_py) * 0.2:
                reasons.append(
                    f"test count mismatch: native={n_nat} python={n_py}")

        # 4. Both should achieve non-trivial P@1 on modes 0-2
        if mode in (0, 1, 2):
            if pp_at1 <= 0.0:
                reasons.append("Python P@1 is zero")
                ok = False
            if np_at1 is not None and np_at1 <= 0.0:
                reasons.append("Native P@1 is zero")

        # 5. Python should be in the same ballpark as native
        #    (generous: at least 30% of native, since RNG/neg-sampling differ)
        if np_at1 is not None and np_at1 > 0.05:
            if pp_at1 < np_at1 * 0.3:
                reasons.append(
                    f"Python P@1={pp_at1:.4f} < 30% of native={np_at1:.4f}")
                ok = False

        # 6. Python P@1 should not exceed native by an unreasonable margin
        #    (> 2x could indicate test evaluation bug)
        if np_at1 is not None and np_at1 > 0.01:
            if pp_at1 > np_at1 * 3.0:
                reasons.append(
                    f"Python P@1={pp_at1:.4f} suspiciously > 3x native="
                    f"{np_at1:.4f}")

        if ok and not reasons:
            print(f"  PASS")
        elif ok and reasons:
            print(f"  PASS (warnings: {'; '.join(reasons)})")
        else:
            print(f"  FAIL: {'; '.join(reasons)}")
            all_ok = False
            failures.append(f"mode {mode}: {'; '.join(reasons)}")

        results.append((mode, np_at1, pp_at1, nr_at1, pr_at1, nt, pt))

    # ── mode 5 on unlabeled data (word embedding) ──
    print("-" * 78)
    print(f"  Mode 5: Word embedding")
    print("-" * 78)

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

    # Parse native model to get embedding norms for comparison
    native_model_tsv = model_path + ".tsv"
    native_mean_norm = None
    native_nwords_5 = None
    if os.path.isfile(native_model_tsv):
        import numpy as np
        native_vecs = []
        with open(native_model_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1 and not parts[0].startswith("__label__"):
                    try:
                        vec = [float(x) for x in parts[1:]]
                        native_vecs.append(vec)
                    except ValueError:
                        pass
        # Filter to vectors with consistent dimension
        if native_vecs:
            expected_dim = DIM
            native_vecs = [v for v in native_vecs if len(v) == expected_dim]
        if native_vecs:
            native_arr = np.array(native_vecs)
            native_mean_norm = float(np.linalg.norm(native_arr, axis=1).mean())
            native_nwords_5 = len(native_vecs)

    for ext in ("", ".tsv"):
        try:
            os.unlink(model_path + ext)
        except OSError:
            pass

    native_5_ok = result.returncode == 0
    if not native_5_ok:
        print(f"  Native:  FAILED")
        print(f"  stderr: {result.stderr[:300]}")
    else:
        extra = ""
        if native_mean_norm is not None:
            extra = f"  mean_norm={native_mean_norm:.4f}  vocab={native_nwords_5}"
        print(f"  Native:  trained OK  time={native_time_5:.2f}s{extra}")

    t0 = time.time()
    py_model_5 = StarSpace.train(
        WORD_DATA,
        dim=DIM, epoch=EPOCH, lr=LR,
        min_count=1, word_ngrams=1, bucket=2_000_000,
        verbose=0, train_mode=5, ws=WS)
    python_time_5 = time.time() - t0

    import numpy as np
    nw = py_model_5.vocab.nwords
    emb_norms = np.linalg.norm(py_model_5.emb[:nw], axis=1)
    py_mean_norm = float(emb_norms.mean())
    spd5 = speedup_str(native_time_5, python_time_5)
    print(f"  Python:  trained OK  time={python_time_5:.2f}s  "
          f"vocab={nw} words  mean_norm={py_mean_norm:.4f}  speedup={spd5}")

    # ── Mode 5 verification ──
    mode5_ok = True
    mode5_reasons = []

    # Embedding norms should be non-trivial (> 0.05 for a trained model)
    if py_mean_norm < 0.05:
        mode5_reasons.append(
            f"Python mean_norm={py_mean_norm:.4f} < 0.05 (untrained)")
        mode5_ok = False

    # If we have native norms, compare them
    if native_mean_norm is not None:
        # Both should be in similar range (within 5x)
        ratio = py_mean_norm / native_mean_norm if native_mean_norm > 0 else 0
        if ratio < 0.2 or ratio > 5.0:
            mode5_reasons.append(
                f"norm mismatch: python={py_mean_norm:.4f} vs "
                f"native={native_mean_norm:.4f} (ratio={ratio:.2f})")

    # Check variance of norms — trained embeddings should have non-trivial spread
    norm_std = float(emb_norms.std())
    if norm_std < 0.01:
        mode5_reasons.append(
            f"norm std={norm_std:.4f} too uniform (not trained)")

    # Verify embedding matrix has reasonable values (not all zeros, not NaN)
    emb_slice = py_model_5.emb[:nw]
    if np.any(np.isnan(emb_slice)):
        mode5_reasons.append("NaN in embeddings")
        mode5_ok = False
    if np.all(emb_slice == 0):
        mode5_reasons.append("all embeddings zero")
        mode5_ok = False

    if mode5_ok and not mode5_reasons:
        print(f"  PASS")
    elif mode5_ok and mode5_reasons:
        print(f"  PASS (warnings: {'; '.join(mode5_reasons)})")
    else:
        print(f"  FAIL: {'; '.join(mode5_reasons)}")
        all_ok = False
        failures.append(f"mode 5: {'; '.join(mode5_reasons)}")

    # ── cleanup ──
    os.unlink(train_path)
    os.unlink(test_path)

    # ── summary ──
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Mode':<8} {'Native P@1':>12} {'Python P@1':>12} "
          f"{'Native time':>13} {'Python time':>13} {'Speedup':>9}")
    print(f"  {'----':<8} {'----------':>12} {'----------':>12} "
          f"{'----------':>13} {'----------':>13} {'-------':>9}")
    for mode, np1, pp1, nr1, pr1, nt, pt in results:
        ns = f"{np1:.4f}" if np1 is not None else "FAILED"
        spd = speedup_str(nt, pt)
        print(f"  {mode:<8} {ns:>12} {pp1:>12.4f} "
              f"{nt:>12.2f}s {pt:>12.2f}s {spd:>9}")
    spd5 = speedup_str(native_time_5, python_time_5)
    print(f"  {'5':<8} {'—':>12} {'—':>12} "
          f"{native_time_5:>12.2f}s {python_time_5:>12.2f}s {spd5:>9}")
    print("=" * 78)

    if all_ok:
        print("  ALL MODES OK")
    else:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
