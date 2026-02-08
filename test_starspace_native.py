"""Native vs Python comparison tests — all training modes.

Both implementations trained on tagged_post.txt with identical params;
vocab, embeddings, P@1, and semantic predictions compared.
"""

import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from starspace import StarSpace, iter_lines

ROOT = Path(__file__).resolve().parent
NATIVE_BIN = str(ROOT / "starspace")
TAGGED_DATA = str(ROOT / "python" / "test" / "tagged_post.txt")

assert os.path.isfile(NATIVE_BIN), \
    f"native binary not found: {NATIVE_BIN} (build with: make BOOST_DIR=... opt)"

# Comparison params — bucket=0 so TSV evaluation matches Python predict()
CMP_DIM = 30
CMP_EPOCH = 10
CMP_LR = 0.01
CMP_BUCKET = 0


# ── helpers ─────────────────────────────────────────────────────────────────

def _native_train(train_path, model_path, *, mode=0, dim=CMP_DIM,
                  epoch=CMP_EPOCH, lr=CMP_LR, ws=5, ngrams=1,
                  bucket=CMP_BUCKET, min_count=1, neg_search_limit=50,
                  margin=0.05, max_neg_samples=10, thread=1, verbose=0,
                  label="__label__", init_rand_sd=0.001, timeout=120):
    cmd = [
        NATIVE_BIN, "train",
        "-trainFile", train_path,
        "-model", model_path,
        "-trainMode", str(mode),
        "-dim", str(dim),
        "-epoch", str(epoch),
        "-lr", str(lr),
        "-ws", str(ws),
        "-negSearchLimit", str(neg_search_limit),
        "-margin", str(margin),
        "-maxNegSamples", str(max_neg_samples),
        "-similarity", "cosine",
        "-adagrad", "1",
        "-shareEmb", "1",
        "-bucket", str(bucket),
        "-ngrams", str(ngrams),
        "-minCount", str(min_count),
        "-verbose", str(verbose),
        "-thread", str(thread),
        "-initRandSd", str(init_rand_sd),
        "-label", label,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"native train failed: {r.stderr}"


def _load_native_tsv(tsv_path):
    result = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[0]:
                try:
                    result[parts[0]] = np.array(
                        [float(x) for x in parts[1:]], dtype=np.float32)
                except ValueError:
                    pass
    return result


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


def _tsv_label_matrix(nat_tsv):
    """Build normalized label embedding matrix from TSV."""
    labels = {}
    words = {}
    for w, v in nat_tsv.items():
        if w.startswith("__label__"):
            labels[w] = v
        else:
            words[w] = v
    label_names = list(labels.keys())
    if not label_names:
        return words, label_names, None
    label_mat = np.array([labels[name] for name in label_names])
    label_norms = np.linalg.norm(label_mat, axis=1, keepdims=True)
    label_mat = label_mat / np.maximum(label_norms, 1e-10)
    return words, label_names, label_mat


def _tsv_p1(nat_tsv, test_path):
    """Compute P@1 using native TSV embeddings, mode-0 style."""
    words, label_names, label_mat = _tsv_label_matrix(nat_tsv)
    if label_mat is None:
        return 0, 0.0
    n = 0
    hits = 0
    for tokens in iter_lines(test_path):
        true_labels = {t for t in tokens if t.startswith("__label__")}
        text_words = [t for t in tokens
                      if not t.startswith("__label__")]
        if not true_labels or not text_words:
            continue
        vecs = [words[w] for w in text_words if w in words]
        if not vecs:
            continue
        lhs = np.sum(vecs, axis=0)
        ln = np.linalg.norm(lhs)
        if ln < 1e-10:
            continue
        lhs = lhs / ln
        scores = label_mat @ lhs
        best_idx = int(np.argmax(scores))
        if label_names[best_idx] in true_labels:
            hits += 1
        n += 1
    return n, hits / max(n, 1)


# ── semantic prediction helpers ─────────────────────────────────────────────

# In-domain queries for tagged_post.txt (political hashtags).
SEMANTIC_QUERIES = {
    "jobs economy unemployment":  "jobs",
    "immigration reform border":  "immigration",
    "tax irs refund":             "irs",
    "benghazi investigation":     "benghazi",
}
SEMANTIC_K = 10
SEMANTIC_MIN_HITS = 3


def _check_semantic_predictions(model):
    """Return (n_hits, n_total, details) for SEMANTIC_QUERIES on model."""
    hits = 0
    details = []
    for text, expected in SEMANTIC_QUERIES.items():
        preds = model.predict(text, k=SEMANTIC_K)
        topk = [lbl.replace("__label__", "") for lbl, _ in preds]
        ok = expected in topk
        if ok:
            hits += 1
        details.append(
            f"  {'OK' if ok else 'MISS':4s} "
            f"'{text}' -> expect '{expected}' in {topk}")
    return hits, len(SEMANTIC_QUERIES), details


def _check_semantic_tsv(nat_tsv):
    """Same as _check_semantic_predictions but using native TSV embeddings."""
    words, label_names, label_mat = _tsv_label_matrix(nat_tsv)
    if label_mat is None:
        return 0, len(SEMANTIC_QUERIES), ["  no labels in TSV"]
    hits = 0
    details = []
    for text, expected in SEMANTIC_QUERIES.items():
        tokens = text.split()
        vecs = [words[w] for w in tokens if w in words]
        if not vecs:
            details.append(
                f"  MISS '{text}' -> expect '{expected}' (no word vecs)")
            continue
        lhs = np.sum(vecs, axis=0)
        ln = np.linalg.norm(lhs)
        if ln < 1e-10:
            details.append(
                f"  MISS '{text}' -> expect '{expected}' (zero norm)")
            continue
        lhs = lhs / ln
        scores = label_mat @ lhs
        top_idx = np.argsort(scores)[::-1][:SEMANTIC_K]
        topk = [label_names[i].replace("__label__", "") for i in top_idx]
        ok = expected in topk
        if ok:
            hits += 1
        details.append(
            f"  {'OK' if ok else 'MISS':4s} "
            f"'{text}' -> expect '{expected}' in {topk}")
    return hits, len(SEMANTIC_QUERIES), details


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tagged_split():
    """80/20 split of tagged_post.txt."""
    train, test = _split_file(TAGGED_DATA)
    yield train, test
    os.unlink(train)
    os.unlink(test)


# Per-mode hyperparams (overrides of CMP_* defaults)
MODE_PARAMS = {
    0: dict(neg_search_limit=5),
    1: dict(neg_search_limit=50, max_neg_samples=10),
    2: dict(lr=0.05, neg_search_limit=5, max_neg_samples=3),
    3: dict(lr=0.05, neg_search_limit=10),
    4: dict(),
}


@pytest.fixture(scope="module", params=sorted(MODE_PARAMS.keys()),
                ids=[f"mode{m}" for m in sorted(MODE_PARAMS.keys())])
def models(request, tagged_split):
    mode = request.param
    train, test = tagged_split
    overrides = MODE_PARAMS[mode]
    lr = overrides.get("lr", CMP_LR)
    nsl = overrides.get("neg_search_limit", 50)
    mns = overrides.get("max_neg_samples", 10)

    nat_path = tempfile.mktemp(prefix=f"ss_m{mode}_")
    _native_train(train, nat_path, mode=mode, lr=lr,
                  neg_search_limit=nsl, max_neg_samples=mns)
    nat_tsv = _load_native_tsv(nat_path + ".tsv")

    py = StarSpace.train(
        train, dim=CMP_DIM, epoch=CMP_EPOCH, lr=lr,
        train_mode=mode, bucket=CMP_BUCKET, verbose=0,
        neg_search_limit=nsl, max_neg_samples=mns)

    yield py, nat_tsv, test
    for ext in ("", ".tsv"):
        try:
            os.unlink(nat_path + ext)
        except OSError:
            pass


# ── tests ───────────────────────────────────────────────────────────────────

class TestVocab:
    def test_word_set(self, models):
        py, nat_tsv, _ = models
        py_words = set(py.vocab.words)
        nat_words = {w for w in nat_tsv if not w.startswith("__label__")}
        assert py_words == nat_words

    def test_label_set(self, models):
        py, nat_tsv, _ = models
        py_labels = set(py.vocab.labels)
        nat_labels = {w for w in nat_tsv if w.startswith("__label__")}
        assert py_labels == nat_labels

    def test_vocab_counts(self, models):
        py, nat_tsv, _ = models
        nat_words = {w for w in nat_tsv if not w.startswith("__label__")}
        nat_labels = {w for w in nat_tsv if w.startswith("__label__")}
        assert py.vocab.nwords == len(nat_words)
        assert py.vocab.nlabels == len(nat_labels)


class TestPredictions:
    def test_semantic_predictions(self, models):
        """In-domain queries return expected labels in top-k."""
        py, _, _ = models
        hits, total, details = _check_semantic_predictions(py)
        assert hits >= SEMANTIC_MIN_HITS, (
            f"{hits}/{total} hits (need {SEMANTIC_MIN_HITS}):\n"
            + "\n".join(details))

    def test_semantic_predictions_native(self, models):
        """Native TSV embeddings also produce expected labels in top-k."""
        _, nat_tsv, _ = models
        hits, total, details = _check_semantic_tsv(nat_tsv)
        assert hits >= SEMANTIC_MIN_HITS, (
            f"{hits}/{total} hits (need {SEMANTIC_MIN_HITS}):\n"
            + "\n".join(details))


class TestEmbeddings:
    def test_p1_comparable(self, models):
        """P@1 within 5 pp (TSV-based evaluation, same method)."""
        py, nat_tsv, test_path = models
        _, py_p1, _ = py.test(test_path, k=1)
        _, nat_p1 = _tsv_p1(nat_tsv, test_path)
        assert py_p1 > 0.01, f"Python P@1={py_p1:.4f}"
        assert nat_p1 > 0.01, f"Native P@1={nat_p1:.4f}"
        assert abs(py_p1 - nat_p1) < 0.05, (
            f"P@1 diff={abs(py_p1 - nat_p1):.4f} "
            f"(py={py_p1:.4f} nat={nat_p1:.4f})")

    def test_embedding_norms(self, models):
        """Mean word embedding norm ratio within 0.5x-2.0x."""
        py, nat_tsv, _ = models
        py_norms, nat_norms = [], []
        for i, w in enumerate(py.vocab.words):
            if w in nat_tsv:
                py_norms.append(np.linalg.norm(py.emb[i]))
                nat_norms.append(np.linalg.norm(nat_tsv[w]))
        assert len(py_norms) > 100
        ratio = np.mean(py_norms) / max(np.mean(nat_norms), 1e-10)
        assert 0.5 < ratio < 2.0, f"norm ratio {ratio:.2f}"

    def test_label_norms(self, models):
        """Mean label embedding norm ratio within 0.5x-2.0x."""
        py, nat_tsv, _ = models
        nw = py.vocab.nwords
        py_norms, nat_norms = [], []
        for i, lbl in enumerate(py.vocab.labels):
            if lbl in nat_tsv:
                py_norms.append(np.linalg.norm(py.emb[nw + i]))
                nat_norms.append(np.linalg.norm(nat_tsv[lbl]))
        assert len(py_norms) > 50
        ratio = np.mean(py_norms) / max(np.mean(nat_norms), 1e-10)
        assert 0.5 < ratio < 2.0, f"label norm ratio {ratio:.2f}"
