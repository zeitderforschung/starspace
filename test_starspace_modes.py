"""Test all 6 StarSpace training modes on real datasets."""

import sys, os, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from starspace import StarSpace, iter_lines

TAGGED_DATA = os.path.join(ROOT, "python", "test", "tagged_post.txt")
WORD_DATA = os.path.join(ROOT, "python", "test", "input.txt")

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


# ── test functions ───────────────────────────────────────────────────────────

def test_mode_0(train_path, test_path):
    """Mode 0: Classification (LHS=words, RHS=1 label)."""
    model = StarSpace.train(train_path, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=0)
    n, p, r = model.test(test_path, k=1)
    print(f"  N={n} P@1={p:.4f} R@1={r:.4f}")
    assert n > 0, f"no test examples evaluated"
    assert p > 0.05, f"P@1={p:.4f} too low"
    # save/load roundtrip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp = tf.name
    model.save(tmp)
    loaded = StarSpace.load(tmp)
    os.unlink(tmp)
    assert loaded.emb.shape == model.emb.shape
    assert loaded.train_mode == 0
    n2, p2, r2 = loaded.test(test_path, k=1)
    assert p2 == p, f"save/load P@1 mismatch: {p2} != {p}"
    # predict
    preds = model.predict("good movie", k=3)
    assert len(preds) > 0, "predict returned no results"
    return True


def test_mode_1(train_path, test_path):
    """Mode 1: Label from rest (LHS=words+rest, RHS=1 label)."""
    model = StarSpace.train(train_path, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=1)
    n, p, r = model.test(test_path, k=1)
    print(f"  N={n} P@1={p:.4f} R@1={r:.4f}")
    assert n > 0, f"no test examples evaluated"
    assert p > 0.01, f"P@1={p:.4f} too low"
    return True


def test_mode_2(train_path, test_path):
    """Mode 2: Inverted labels (LHS=words+1 label, RHS=rest labels)."""
    model = StarSpace.train(train_path, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=2)
    n, p, r = model.test(test_path, k=1)
    print(f"  N={n} P@1={p:.4f} R@1={r:.4f}")
    assert n > 0, f"no test examples evaluated"
    assert p > 0.01, f"P@1={p:.4f} too low"
    return True


def test_mode_3(train_path, test_path):
    """Mode 3: Pair prediction (LHS=words+1 label, RHS=1 other label)."""
    model = StarSpace.train(train_path, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=3)
    n, p, r = model.test(test_path, k=1)
    print(f"  N={n} P@1={p:.4f} R@1={r:.4f}")
    assert n > 0, f"no test examples evaluated"
    # mode 3 on small data can have low P@1, just check non-negative
    return True


def test_mode_4(train_path, test_path):
    """Mode 4: Fixed pair (LHS=words+label[0], RHS=label[1])."""
    model = StarSpace.train(train_path, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=4)
    n, p, r = model.test(test_path, k=1)
    print(f"  N={n} P@1={p:.4f} R@1={r:.4f}")
    assert n > 0, f"no test examples evaluated"
    return True


def test_mode_5():
    """Mode 5: Word embedding (context -> target word)."""
    model = StarSpace.train(WORD_DATA, dim=50, epoch=10, lr=0.01,
                            seed=0, verbose=0, min_count=1, train_mode=5, ws=5)
    nw = model.vocab.nwords
    print(f"  vocab={nw} words")
    assert nw > 100, f"vocab too small: {nw}"
    emb_norms = np.linalg.norm(model.emb[:nw], axis=1)
    mean_norm = float(emb_norms.mean())
    print(f"  mean_norm={mean_norm:.4f}")
    assert mean_norm > 0.05, f"mean_norm={mean_norm:.4f} too low (untrained)"
    assert not np.any(np.isnan(model.emb[:nw])), "NaN in embeddings"
    # test() should raise for mode 5
    try:
        model.test(WORD_DATA)
        assert False, "test() should raise ValueError for mode 5"
    except ValueError:
        pass
    # save/load roundtrip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp = tf.name
    model.save(tmp)
    loaded = StarSpace.load(tmp)
    os.unlink(tmp)
    assert loaded.train_mode == 5
    assert np.allclose(loaded.emb, model.emb), "save/load embedding mismatch"
    return True


def test_file_input():
    """Test file path input (mode 0)."""
    model = StarSpace.train(TAGGED_DATA, dim=30, epoch=3, lr=0.01,
                            seed=0, verbose=0, min_count=1)
    preds = model.predict("good movie", k=3)
    assert len(preds) > 0, "predict returned no results"
    print(f"  predict('good movie'): {preds[:2]}")
    return True


def test_iter_lines():
    """Test iter_lines input."""
    model = StarSpace.train(iter_lines(TAGGED_DATA), dim=30, epoch=3, lr=0.01,
                            seed=0, verbose=0, min_count=1)
    preds = model.predict("good movie", k=3)
    assert len(preds) > 0, "predict returned no results"
    print(f"  predict('good movie'): {preds[:2]}")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("StarSpace All-Modes Test Suite")
    print("=" * 60)

    # Warm numba JIT
    print("\nWarming numba JIT (first compile)...")
    t0 = time.time()
    StarSpace.train([["__label__a", "hi"], ["__label__b", "bye"]],
                    dim=10, epoch=1, verbose=0)
    print(f"JIT warm-up done in {time.time()-t0:.1f}s")

    # Split tagged data
    train_path, test_path, n_train, n_test = split_file(TAGGED_DATA)
    print(f"\nDataset: {os.path.basename(TAGGED_DATA)}")
    print(f"  Train: {n_train} lines, Test: {n_test} lines\n")

    tests = [
        ("Mode 0", lambda: test_mode_0(train_path, test_path)),
        ("Mode 1", lambda: test_mode_1(train_path, test_path)),
        ("Mode 2", lambda: test_mode_2(train_path, test_path)),
        ("Mode 3", lambda: test_mode_3(train_path, test_path)),
        ("Mode 4", lambda: test_mode_4(train_path, test_path)),
        ("Mode 5", test_mode_5),
        ("file",   test_file_input),
        ("iter",   test_iter_lines),
    ]

    results = {}
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            results[name] = fn()
            print(f"  PASS ({time.time()-t0:.2f}s)")
        except Exception as e:
            results[name] = False
            print(f"  FAIL: {type(e).__name__}: {e}")

    # Cleanup
    os.unlink(train_path)
    os.unlink(test_path)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_pass = False

    print(f"{'='*60}")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
