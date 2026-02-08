"""StarSpace self-consistency tests — determinism, save/load, API, convergence.

Python-only tests that don't compare against the native binary.

Usage:
    python3 -m pytest test_starspace.py -v
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from starspace import StarSpace, iter_lines

ROOT = Path(__file__).resolve().parent
TAGGED_DATA = str(ROOT / "python" / "test" / "tagged_post.txt")

DIM = 10
EPOCH = 3
LR = 0.01
BUCKET = 100


def _split_file(path, train_frac=0.8, seed=42):
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
    return train_f.name, test_f.name


@pytest.fixture(scope="module")
def tagged_split():
    """80/20 split of tagged_post.txt."""
    train, test = _split_file(TAGGED_DATA)
    yield train, test
    os.unlink(train)
    os.unlink(test)


class TestDeterminism:

    def test_same_seed_same_output(self):
        """Bit-identical embeddings across runs with same seed."""
        models = []
        for _ in range(2):
            m = StarSpace.train(
                TAGGED_DATA, dim=DIM, epoch=EPOCH, lr=LR,
                seed=0, verbose=0, bucket=BUCKET)
            models.append(m)
        assert np.array_equal(models[0].emb, models[1].emb)
        assert models[0].vocab.words == models[1].vocab.words
        assert models[0].vocab.labels == models[1].vocab.labels

    def test_save_load_roundtrip(self):
        """Save/load preserves embeddings, vocab, and hyperparams."""
        train_path = tempfile.mktemp(suffix=".txt")
        model_path = tempfile.mktemp(suffix=".npz")
        try:
            with open(train_path, "w") as f:
                for _ in range(50):
                    f.write("hello world __label__a\n")
                    f.write("goodbye moon __label__b\n")

            orig = StarSpace.train(
                train_path, dim=20, epoch=3, verbose=0, train_mode=0)
            orig.save(model_path)
            loaded = StarSpace.load(model_path)

            assert np.array_equal(orig.emb, loaded.emb)
            assert orig.vocab.words == loaded.vocab.words
            assert orig.vocab.labels == loaded.vocab.labels
            for attr in ("dim", "word_ngrams", "margin",
                         "neg_search_limit", "lr", "epoch", "seed",
                         "norm_limit", "train_mode",
                         "max_neg_samples", "batch_size"):
                assert getattr(orig, attr) == getattr(loaded, attr), attr

            pred_orig = orig.predict("hello world", k=2)
            pred_load = loaded.predict("hello world", k=2)
            assert pred_orig == pred_load
        finally:
            for p in (train_path, model_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass


class TestConvergence:

    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
    def test_supervised_converges(self, mode, tagged_split):
        """Each supervised mode trains and achieves P@1 > 0."""
        train, test = tagged_split
        m = StarSpace.train(
            train, dim=DIM, epoch=EPOCH, lr=LR, verbose=0,
            train_mode=mode, bucket=BUCKET)
        n, p, _ = m.test(test, k=1)
        assert n > 0, f"mode {mode}: no test examples"
        assert p > 0, f"mode {mode}: P@1=0 (model didn't learn)"


class TestAPI:

    def test_file_input(self):
        """Training from file path produces predictions."""
        m = StarSpace.train(
            TAGGED_DATA, dim=DIM, epoch=EPOCH, lr=LR,
            seed=0, verbose=0, bucket=BUCKET)
        preds = m.predict("good movie", k=3)
        assert len(preds) > 0
        assert all(lbl.startswith("__label__") for lbl, _ in preds)

    def test_iter_input(self):
        """Training from iterable produces predictions."""
        m = StarSpace.train(
            iter_lines(TAGGED_DATA), dim=DIM, epoch=EPOCH, lr=LR,
            seed=0, verbose=0, bucket=BUCKET)
        preds = m.predict("good movie", k=3)
        assert len(preds) > 0

    def test_predict_returns_labels(self, tagged_split):
        """predict() returns valid label names and scores."""
        train, _ = tagged_split
        m = StarSpace.train(
            train, dim=DIM, epoch=EPOCH, lr=LR, verbose=0,
            bucket=BUCKET)
        preds = m.predict("the house", k=5)
        assert len(preds) > 0
        for label, score in preds:
            assert label.startswith("__label__")
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0
