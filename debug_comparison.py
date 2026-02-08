"""Diagnostic: collect actual metric values for all modes."""

import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from starspace import StarSpace, iter_lines

ROOT = Path(__file__).resolve().parent
NATIVE_BIN = str(ROOT / "starspace")
TAGGED_DATA = str(ROOT / "python" / "test" / "tagged_post.txt")

CMP_DIM = 30
CMP_EPOCH = 10
CMP_LR = 0.01
CMP_BUCKET = 0


def native_train(train_path, model_path, *, mode=0):
    cmd = [
        NATIVE_BIN, "train",
        "-trainFile", train_path,
        "-model", model_path,
        "-trainMode", str(mode),
        "-dim", str(CMP_DIM),
        "-epoch", str(CMP_EPOCH),
        "-lr", str(CMP_LR),
        "-negSearchLimit", "50",
        "-margin", "0.05",
        "-maxNegSamples", "10",
        "-similarity", "cosine",
        "-adagrad", "1",
        "-shareEmb", "1",
        "-bucket", str(CMP_BUCKET),
        "-ngrams", "1",
        "-minCount", "1",
        "-verbose", "0",
        "-thread", "1",
        "-initRandSd", "0.001",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"native train failed: {r.stderr}"


def load_native_tsv(tsv_path):
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


def split_file(path, train_frac=0.8, seed=42):
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


def tsv_p1(nat_tsv, test_path):
    labels = {}
    words = {}
    for w, v in nat_tsv.items():
        if w.startswith("__label__"):
            labels[w] = v
        else:
            words[w] = v
    label_names = list(labels.keys())
    if not label_names:
        return 0, 0.0
    label_mat = np.array([labels[name] for name in label_names])
    label_norms = np.linalg.norm(label_mat, axis=1, keepdims=True)
    label_mat = label_mat / np.maximum(label_norms, 1e-10)
    n = 0
    hits = 0
    for tokens in iter_lines(test_path):
        true_labels = {t for t in tokens if t.startswith("__label__")}
        text_words = [t for t in tokens if not t.startswith("__label__")]
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


train, test = split_file(TAGGED_DATA)

print(f"{'mode':<6} {'py_p1':>8} {'nat_p1':>8} {'diff':>8} "
      f"{'w_ratio':>8} {'l_ratio':>8} {'py>nat?':>8}")
print("-" * 60)

for mode in range(5):
    nat_path = tempfile.mktemp(prefix=f"ss_dbg{mode}_")
    native_train(train, nat_path, mode=mode)
    nat_tsv = load_native_tsv(nat_path + ".tsv")

    py = StarSpace.train(
        train, dim=CMP_DIM, epoch=CMP_EPOCH, lr=CMP_LR,
        train_mode=mode, bucket=CMP_BUCKET, verbose=0)

    _, py_p1, _ = py.test(test, k=1)
    _, nat_p1 = tsv_p1(nat_tsv, test)

    # Word norm ratio
    py_norms, nat_norms = [], []
    for i, w in enumerate(py.vocab.words):
        if w in nat_tsv:
            py_norms.append(np.linalg.norm(py.emb[i]))
            nat_norms.append(np.linalg.norm(nat_tsv[w]))
    w_ratio = np.mean(py_norms) / max(np.mean(nat_norms), 1e-10)

    # Label norm ratio
    nw = py.vocab.nwords
    py_ln, nat_ln = [], []
    for i, lbl in enumerate(py.vocab.labels):
        if lbl in nat_tsv:
            py_ln.append(np.linalg.norm(py.emb[nw + i]))
            nat_ln.append(np.linalg.norm(nat_tsv[lbl]))
    l_ratio = np.mean(py_ln) / max(np.mean(nat_ln), 1e-10)

    print(f"{mode:<6} {py_p1:>8.4f} {nat_p1:>8.4f} {py_p1 - nat_p1:>+8.4f} "
          f"{w_ratio:>8.3f} {l_ratio:>8.3f} "
          f"{'YES' if py_p1 > nat_p1 else 'no':>8}")

    for ext in ("", ".tsv"):
        try:
            os.unlink(nat_path + ext)
        except OSError:
            pass

os.unlink(train)
os.unlink(test)
