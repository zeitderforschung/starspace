"""starspace — StarSpace "Embed All The Things" in pure Python.

Implements supervised training modes (trainMode 0-4) with:
- Hinge (margin ranking) loss + negative sampling
- Cosine similarity (L2-normalized embeddings)
- AdaGrad optimisation
- Shared LHS/RHS embedding matrix
- Word n-grams

Training modes::

    0  Classification   — LHS = words,             RHS = 1 random label
    1  Label from rest  — LHS = words + rest labels, RHS = 1 random label
    2  Inverted labels  — LHS = words + 1 label,   RHS = rest labels
    3  Pair prediction  — LHS = words + 1 label,   RHS = 1 other label
    4  Fixed pair       — LHS = words + label[0],  RHS = label[1]

::

    model = StarSpace.train("train.txt", dim=100, epoch=5)
    model.test("test.txt")                     # → (N, P@1, R@1)
    model.predict("the food was great")        # → [("__label__pos", 0.92)]

    # From any iterable of token lists (spills to temp file):
    lines = [["__label__pos", "love", "this"], ["__label__neg", "awful"]]
    model = StarSpace.train(lines, dim=100, epoch=5)

Requires only **numpy** and **numba** (no C compiler, no scipy).

Memory footprint for a 10 GB text file (~5 M unique words, dim=100):
  text mmap        0 B  (OS pages from disk)
  hash table    ~600 MB (6 arrays × ~20 M slots)
  remap          ~80 MB
  embeddings   ~2.8 GB  ((5 M + labels + buckets) × 100 × 4)
  adagrad       ~28 MB
  total        ~3.5 GB  — the text is never copied into RAM
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numba import njit


# ── public helpers ────────────────────────────────────────────────────────────

def iter_lines(path: str) -> Iterator[list[str]]:
    """Yield tokenized lines from a text file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            tokens = line.split()
            if tokens:
                yield tokens


# ── deterministic hash (matches C++ StarSpace / fasttext) ────────────────────


@njit(cache=True)
def _fnv1a_bytes(data):
    """FNV-1a 32-bit over a uint8 array with signed-char XOR."""
    h = np.uint32(2166136261)
    for i in range(len(data)):
        b = data[i]
        sb = np.uint32(b) if b < 128 else np.uint32(np.int32(np.int8(b)))
        h = (h ^ sb) * np.uint32(16777619)
    return np.int32(h)

# ── mmap text pipeline ────────────────────────────────────────────────────────
#
# Operates directly on a memory-mapped text file (uint8 bytes):
#   Pass 1: _vocab_scan     — hash table from raw bytes
#   Pass 2: _extract_vocab  — filter, sort, assign final IDs → Vocab
#   Train:  _train_epoch    — retokenise from mmap each epoch, train inline


@njit(cache=True)
def _vocab_scan(buf, buf_len,
                ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                ht_freq, ht_is_label,
                table_mask, label_prefix):
    """Pass 1: scan mmap bytes → hash table.

    Returns (n_unique, n_tokens, n_lines, status).
    status: 0 = success, -1 = table overflow (caller should grow & retry).
    """
    n_unique = np.int64(0)
    n_tokens = np.int64(0)
    n_lines  = np.int64(0)
    table_size = np.int64(table_mask + 1)
    lp_len = len(label_prefix)

    i = np.int64(0)
    while i < buf_len:
        b = buf[i]
        if b == 32 or b == 9 or b == 13:
            i += 1
            continue
        if b == 10:
            n_lines += 1
            i += 1
            continue

        tok_start = i
        while i < buf_len:
            b2 = buf[i]
            if b2 == 32 or b2 == 9 or b2 == 10 or b2 == 13:
                break
            i += 1
        tok_len = np.int32(i - tok_start)

        # FNV-1a
        h = np.uint32(2166136261)
        for k in range(tok_start, tok_start + tok_len):
            b3 = buf[k]
            sb = np.uint32(b3) if b3 < 128 else np.uint32(np.int32(np.int8(b3)))
            h = (h ^ sb) * np.uint32(16777619)
        fnv = np.int32(h)

        # open-addressing lookup
        slot = np.int64(np.uint32(h) & np.uint32(table_mask))
        while ht_occ[slot] == np.int8(1):
            if ht_fnv[slot] == fnv and ht_tok_len[slot] == tok_len:
                match = True
                ref = ht_tok_start[slot]
                for k in range(tok_len):
                    if buf[tok_start + k] != buf[ref + k]:
                        match = False
                        break
                if match:
                    ht_freq[slot] += np.int64(1)
                    break
            slot = (slot + 1) & table_mask

        if ht_occ[slot] == np.int8(0):
            if n_unique >= table_size * 7 // 10:
                return n_unique, n_tokens, n_lines, np.int32(-1)
            ht_occ[slot] = np.int8(1)
            ht_fnv[slot] = fnv
            ht_tok_start[slot] = tok_start
            ht_tok_len[slot] = tok_len
            ht_freq[slot] = np.int64(1)
            is_label = np.int8(0)
            if tok_len >= lp_len:
                is_label = np.int8(1)
                for k in range(lp_len):
                    if buf[tok_start + k] != label_prefix[k]:
                        is_label = np.int8(0)
                        break
            ht_is_label[slot] = is_label
            n_unique += 1

        n_tokens += 1

    if buf_len > 0 and buf[buf_len - 1] != 10:
        n_lines += 1

    return n_unique, n_tokens, n_lines, np.int32(0)


# ── training helpers ─────────────────────────────────────────────────────────

@njit(cache=True)
def _lookup_token(buf, tok_start, tok_end,
                  ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                  table_mask, remap):
    """Hash table lookup for one token. Returns (remapped_id, fnv_hash)."""
    tok_len = np.int32(tok_end - tok_start)
    h = np.uint32(2166136261)
    for k in range(tok_start, tok_end):
        bb = buf[k]
        sb = np.uint32(bb) if bb < 128 else np.uint32(np.int32(np.int8(bb)))
        h = (h ^ sb) * np.uint32(16777619)
    fnv = np.int32(h)
    slot = np.int64(np.uint32(h) & np.uint32(table_mask))
    while ht_occ[slot] == np.int8(1):
        if ht_fnv[slot] == fnv and ht_tok_len[slot] == tok_len:
            match = True
            ref = ht_tok_start[slot]
            for k in range(tok_len):
                if buf[tok_start + k] != buf[ref + k]:
                    match = False
                    break
            if match:
                return remap[slot], fnv
        slot = (slot + 1) & table_mask
    return np.int32(-1), fnv


@njit(cache=True)
def _build_word_ctx(wids, whash, nw, word_ngrams, bucket,
                    ngram_base, ctx_buf):
    """Build LHS context: word IDs + word n-gram bucket IDs. Returns n_ctx."""
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)
    n = np.int32(0)
    for k in range(nw):
        ctx_buf[n] = wids[k]
        n += 1
    if word_ngrams > 1 and bucket > 0:
        for i in range(nw):
            hv = np.uint64(np.int64(whash[i])) & _M
            for j in range(i + 1, min(nw, i + word_ngrams)):
                hv = (hv * np.uint64(116049371) +
                      (np.uint64(np.int64(whash[j])) & _M)) & _M
                ctx_buf[n] = np.int32(
                    ngram_base + np.int32(hv % np.uint64(bucket)))
                n += 1
    return n


@njit(fastmath=True, cache=True)
def _train_batch(emb, adagrad_lhs, adagrad_rhs,
                 batch_ctx, batch_n_ctx, batch_tgt, batch_n_tgt,
                 n_examples,
                 batch_lhs, batch_rhs_pos, rhs_neg,
                 batch_neg_mean, batch_grad_w,
                 neg_ids, batch_neg_flags, batch_n_valid,
                 dim, nwords, nlabels, margin, neg_search_limit,
                 cur_lr, rng_state, norm_limit, mode,
                 max_neg_samples, neg_pool, neg_pool_size,
                 neg_vecs,
                 ex_labels, ex_offsets, n_multi_ex,
                 neg_label_buf, neg_label_cnt):
    """Train one batch of examples with shared negatives.

    Matches native C++ trainOneBatch (model.cpp):
    1. Project all LHS/RHS+ in the batch
    2. Draw negSearchLimit negatives once (shared across batch)
    3. Per example: check each negative for hinge violation, cap maxNegSamples
    4. Update gradients per example

    Returns (loss, rng_state).
    """
    loss_sum = np.float64(0.0)

    # ── 1. Forward pass: project all LHS and RHS+ ──
    for ex in range(n_examples):
        n_ctx = batch_n_ctx[ex]
        n_tgt = batch_n_tgt[ex]

        # LHS: sum + L2 normalise
        for d in range(dim):
            batch_lhs[ex, d] = np.float32(0.0)
        for k in range(n_ctx):
            row = batch_ctx[ex, k]
            for d in range(dim):
                batch_lhs[ex, d] += emb[row, d]
        norm_sq = np.float32(0.0)
        for d in range(dim):
            norm_sq += batch_lhs[ex, d] * batch_lhs[ex, d]
        inv = np.float32(1.0 / np.float32(
            np.sqrt(np.float64(norm_sq)) + 1e-10))
        for d in range(dim):
            batch_lhs[ex, d] *= inv

        # RHS+: sum + L2 normalise
        for d in range(dim):
            batch_rhs_pos[ex, d] = np.float32(0.0)
        for t in range(n_tgt):
            tgt = batch_tgt[ex, t]
            for d in range(dim):
                batch_rhs_pos[ex, d] += emb[tgt, d]
        norm_sq = np.float32(0.0)
        for d in range(dim):
            norm_sq += batch_rhs_pos[ex, d] * batch_rhs_pos[ex, d]
        inv = np.float32(1.0 / np.float32(
            np.sqrt(np.float64(norm_sq)) + 1e-10))
        for d in range(dim):
            batch_rhs_pos[ex, d] *= inv

    # ── 2. Draw negatives once (shared across batch) ──
    for ni in range(neg_search_limit):
        rng_state = (rng_state * np.int64(48271)) % np.int64(2147483647)

        if mode == 2 and n_multi_ex > 0:
            # multi-label negative: pick random example, exclude one label,
            # sum the rest (matches native getRandomRHS for mode 2)
            ex_idx = np.int32(
                np.uint64(rng_state) % np.uint64(n_multi_ex))
            start = ex_offsets[ex_idx]
            end = ex_offsets[ex_idx + 1]
            n_ex_labels = np.int32(end - start)
            rng_state = (rng_state * np.int64(48271)) % np.int64(2147483647)
            exclude_idx = np.int32(
                np.uint64(rng_state) % np.uint64(n_ex_labels))
            # sum embeddings of all labels except excluded
            for d in range(dim):
                neg_vecs[ni, d] = np.float32(0.0)
            cnt = np.int32(0)
            for k in range(n_ex_labels):
                if k != exclude_idx:
                    lid = ex_labels[start + k]
                    neg_label_buf[ni, cnt] = lid
                    cnt += 1
                    for d in range(dim):
                        neg_vecs[ni, d] += emb[lid, d]
            neg_label_cnt[ni] = cnt
            neg_ids[ni] = np.int32(-1)  # sentinel: no single ID
        else:
            # single-label negative
            if neg_pool_size > 0:
                neg_label = neg_pool[np.int32(
                    np.uint64(rng_state) % np.uint64(neg_pool_size))]
            else:
                neg_label = nwords + np.int32(
                    np.uint64(rng_state) % np.uint64(nlabels))
            neg_ids[ni] = neg_label
            neg_label_cnt[ni] = np.int32(0)

            # project: copy single embedding
            for d in range(dim):
                neg_vecs[ni, d] = emb[neg_label, d]

        # L2 normalise projected negative
        norm_sq2 = np.float32(0.0)
        for d in range(dim):
            norm_sq2 += neg_vecs[ni, d] * neg_vecs[ni, d]
        inv2 = np.float32(
            1.0 / np.float32(np.sqrt(np.float64(norm_sq2)) + 1e-10))
        for d in range(dim):
            neg_vecs[ni, d] *= inv2

    # ── 3. Per-example: compute loss and gradients ──
    total_loss = np.float64(0.0)
    any_update = False

    for ex in range(n_examples):
        n_tgt = batch_n_tgt[ex]

        # posSim
        pos_sim = np.float32(0.0)
        for d in range(dim):
            pos_sim += batch_lhs[ex, d] * batch_rhs_pos[ex, d]

        for d in range(dim):
            batch_neg_mean[ex, d] = np.float32(0.0)
        batch_n_valid[ex] = np.int32(0)
        ex_loss = np.float64(0.0)

        for ni in range(neg_search_limit):
            nl = neg_ids[ni]
            # skip if neg == any positive target
            is_pos = False
            for t in range(n_tgt):
                if nl == batch_tgt[ex, t]:
                    is_pos = True
                    break
            if is_pos:
                batch_neg_flags[ex, ni] = np.int32(0)
                continue

            neg_sim = np.float32(0.0)
            for d in range(dim):
                neg_sim += batch_lhs[ex, d] * neg_vecs[ni, d]
            trip = margin - pos_sim + neg_sim
            # C++ model.cpp:390 caps at kMaxLoss = 10e7
            if trip > np.float32(10e7):
                trip = np.float32(10e7)

            if trip > np.float32(0.0):
                for d in range(dim):
                    batch_neg_mean[ex, d] += neg_vecs[ni, d]
                batch_n_valid[ex] += np.int32(1)
                batch_neg_flags[ex, ni] = np.int32(1)
                ex_loss += np.float64(trip)
                if batch_n_valid[ex] >= max_neg_samples:
                    # zero out remaining flags
                    for ni2 in range(ni + 1, neg_search_limit):
                        batch_neg_flags[ex, ni2] = np.int32(0)
                    break
            else:
                batch_neg_flags[ex, ni] = np.int32(0)

        nv = batch_n_valid[ex]
        if nv == 0:
            continue

        any_update = True
        # C++ model.cpp:458 normalizes loss per example by negSearchLimit
        loss_sum += ex_loss / np.float64(neg_search_limit)

        # gradW = mean(negs) - rhs_pos
        for d in range(dim):
            batch_grad_w[ex, d] = (batch_neg_mean[ex, d]
                                   / np.float32(nv)
                                   - batch_rhs_pos[ex, d])

    if not any_update:
        return loss_sum, rng_state

    # ── 4. Backward pass: update per example ──
    for ex in range(n_examples):
        nv = batch_n_valid[ex]
        if nv == 0:
            continue

        n_ctx = batch_n_ctx[ex]
        n_tgt = batch_n_tgt[ex]

        # ||gradW||^2 for LHS AdaGrad
        grad_norm_sq = np.float32(0.0)
        for d in range(dim):
            grad_norm_sq += batch_grad_w[ex, d] * batch_grad_w[ex, d]

        # ||lhs||^2 for RHS AdaGrad
        lhs_norm_sq = np.float32(0.0)
        for d in range(dim):
            lhs_norm_sq += batch_lhs[ex, d] * batch_lhs[ex, d]

        # update LHS
        for k in range(n_ctx):
            row = batch_ctx[ex, k]
            adagrad_lhs[row] += grad_norm_sq / np.float32(dim)
            eff = cur_lr / np.float32(
                np.sqrt(np.float64(adagrad_lhs[row]) + 1e-6))
            for d in range(dim):
                emb[row, d] -= eff * batch_grad_w[ex, d]

        # update RHS positive
        for t in range(n_tgt):
            tgt = batch_tgt[ex, t]
            adagrad_rhs[tgt] += lhs_norm_sq / np.float32(dim)
            eff = cur_lr / np.float32(
                np.sqrt(np.float64(adagrad_rhs[tgt]) + 1e-6))
            for d in range(dim):
                emb[tgt, d] += eff * batch_lhs[ex, d]

        # update RHS negatives
        nr = cur_lr / np.float32(nv)
        for ni in range(neg_search_limit):
            if batch_neg_flags[ex, ni] == np.int32(1):
                if mode == 2 and neg_label_cnt[ni] > 0:
                    # multi-label: update each label in the set
                    for k in range(neg_label_cnt[ni]):
                        nl = neg_label_buf[ni, k]
                        adagrad_rhs[nl] += lhs_norm_sq / np.float32(dim)
                        eff = nr / np.float32(
                            np.sqrt(np.float64(adagrad_rhs[nl]) + 1e-6))
                        for d in range(dim):
                            emb[nl, d] -= eff * batch_lhs[ex, d]
                else:
                    nl = neg_ids[ni]
                    adagrad_rhs[nl] += lhs_norm_sq / np.float32(dim)
                    eff = nr / np.float32(
                        np.sqrt(np.float64(adagrad_rhs[nl]) + 1e-6))
                    for d in range(dim):
                        emb[nl, d] -= eff * batch_lhs[ex, d]

    # norm clipping for all modified rows
    if norm_limit > np.float32(0.0):
        for ex in range(n_examples):
            if batch_n_valid[ex] == 0:
                continue
            for k in range(batch_n_ctx[ex]):
                row = batch_ctx[ex, k]
                rn = np.float32(0.0)
                for d in range(dim):
                    rn += emb[row, d] * emb[row, d]
                rn = np.float32(np.sqrt(np.float64(rn)))
                if rn > norm_limit:
                    sc = norm_limit / rn
                    for d in range(dim):
                        emb[row, d] *= sc
            for t in range(batch_n_tgt[ex]):
                tgt = batch_tgt[ex, t]
                rn = np.float32(0.0)
                for d in range(dim):
                    rn += emb[tgt, d] * emb[tgt, d]
                rn = np.float32(np.sqrt(np.float64(rn)))
                if rn > norm_limit:
                    sc = norm_limit / rn
                    for d in range(dim):
                        emb[tgt, d] *= sc
        # clip neg rows (shared across batch, clip once)
        for ni in range(neg_search_limit):
            any_flag = False
            for ex in range(n_examples):
                if batch_neg_flags[ex, ni] == np.int32(1):
                    any_flag = True
                    break
            if any_flag:
                if mode == 2 and neg_label_cnt[ni] > 0:
                    for k in range(neg_label_cnt[ni]):
                        nl = neg_label_buf[ni, k]
                        rn = np.float32(0.0)
                        for d in range(dim):
                            rn += emb[nl, d] * emb[nl, d]
                        rn = np.float32(np.sqrt(np.float64(rn)))
                        if rn > norm_limit:
                            sc = norm_limit / rn
                            for d in range(dim):
                                emb[nl, d] *= sc
                else:
                    nl = neg_ids[ni]
                    rn = np.float32(0.0)
                    for d in range(dim):
                        rn += emb[nl, d] * emb[nl, d]
                    rn = np.float32(np.sqrt(np.float64(rn)))
                    if rn > norm_limit:
                        sc = norm_limit / rn
                        for d in range(dim):
                            emb[nl, d] *= sc

    return loss_sum, rng_state


# ── line offsets (for per-epoch shuffling) ────────────────────────────────────

@njit(cache=True)
def _find_line_offsets(buf, buf_len):
    """Pre-scan mmap for line start byte offsets."""
    if buf_len == 0:
        return np.empty(0, np.int64)
    n = np.int64(1)
    for i in range(buf_len):
        if buf[i] == 10 and i + 1 < buf_len:
            n += 1
    offsets = np.empty(n, np.int64)
    offsets[0] = np.int64(0)
    idx = np.int64(1)
    for i in range(buf_len):
        if buf[i] == 10 and i + 1 < buf_len:
            offsets[idx] = np.int64(i + 1)
            idx += 1
    return offsets


# ── training epoch (retokenises from mmap each pass) ─────────────────────────

@njit(fastmath=True, cache=True)
def _train_epoch(emb, adagrad_lhs, adagrad_rhs, buf, buf_len,
                 ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                 table_mask, remap,
                 nwords, nlabels, dim, word_ngrams, bucket,
                 margin, neg_search_limit, start_rate, finish_rate,
                 rng_state, norm_limit, mode,
                 line_offsets, n_lines, perm,
                 max_neg_samples, neg_pool, neg_pool_size,
                 batch_size,
                 ex_labels, ex_offsets, n_multi_ex,
                 n_train_ex):
    """One training epoch: shuffle lines, retokenise, train.

    LR schedule matches native C++: stepwise decay every 1000 samples
    from start_rate to finish_rate within the epoch.

    batch_size: number of examples per batch (native default=5).
    Examples in a batch share the same set of negative samples.

    Returns (loss_sum, n_steps, rng_state).
    """
    loss_sum = np.float64(0.0)
    n_steps = np.int32(0)
    ngram_base = nwords + nlabels

    # stepwise LR decay every 1000 samples (matches native kDecrStep)
    # n_train_ex = C++ numSamples (filtered examples after parser.check())
    K_DECR_STEP = np.int64(1000)
    n_k_steps = max(n_train_ex // K_DECR_STEP, np.int64(1))
    decr_per_k = np.float32(
        (np.float64(start_rate) - np.float64(finish_rate))
        / np.float64(n_k_steps))
    cur_lr = start_rate
    sample_count = np.int64(0)

    # per-line scratch
    MAX_TOK = np.int32(10000)
    line_wids  = np.empty(MAX_TOK, np.int32)
    line_whash = np.empty(MAX_TOK, np.int32)
    line_lids  = np.empty(MAX_TOK, np.int32)

    max_ctx = MAX_TOK * max(word_ngrams, np.int32(1)) + MAX_TOK
    ctx_buf    = np.empty(max_ctx, np.int32)
    target_buf = np.empty(MAX_TOK, np.int32)

    # batch scratch arrays
    BS = batch_size
    batch_ctx      = np.empty((BS, max_ctx), np.int32)
    batch_n_ctx    = np.empty(BS, np.int32)
    batch_tgt      = np.empty((BS, MAX_TOK), np.int32)
    batch_n_tgt    = np.empty(BS, np.int32)
    batch_lhs      = np.empty((BS, dim), np.float32)
    batch_rhs_pos  = np.empty((BS, dim), np.float32)
    rhs_neg        = np.empty(dim, np.float32)
    batch_neg_mean = np.empty((BS, dim), np.float32)
    batch_grad_w   = np.empty((BS, dim), np.float32)
    neg_ids_b      = np.empty(neg_search_limit, np.int32)
    batch_neg_flags = np.empty((BS, neg_search_limit), np.int32)
    batch_n_valid  = np.empty(BS, np.int32)
    neg_vecs       = np.empty((neg_search_limit, dim), np.float32)
    n_buffered     = np.int32(0)

    # mode 2 multi-label negative scratch
    max_neg_per_ex = np.int32(0)
    if n_multi_ex > 0:
        for i in range(n_multi_ex):
            sz = np.int32(ex_offsets[i + 1] - ex_offsets[i] - 1)
            if sz > max_neg_per_ex:
                max_neg_per_ex = sz
    neg_label_buf = np.empty(
        (neg_search_limit, max(max_neg_per_ex, np.int32(1))), np.int32)
    neg_label_cnt = np.empty(neg_search_limit, np.int32)

    # Fisher-Yates shuffle (matches native C++ random_shuffle per epoch)
    for i in range(n_lines - 1, 0, -1):
        rng_state = (rng_state * np.int64(48271)) % np.int64(2147483647)
        j = np.int64(np.uint64(rng_state) % np.uint64(i + 1))
        perm[i], perm[j] = perm[j], perm[i]

    for li in range(n_lines):
        # find line bounds
        line_start = line_offsets[perm[li]]
        line_end = line_start
        while line_end < buf_len and buf[line_end] != np.uint8(10):
            line_end += 1

        # tokenise this line
        nw_line = np.int32(0)
        nl_line = np.int32(0)
        in_token = False
        tok_start = line_start
        pos = line_start

        while pos <= line_end:
            if pos < line_end:
                b = buf[pos]
            else:
                b = np.uint8(32)  # virtual space to flush last token

            if b == 32 or b == 9 or b == 13 or b == 10:
                if in_token:
                    fid, fnv = _lookup_token(
                        buf, tok_start, pos,
                        ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                        table_mask, remap)
                    if fid >= 0:
                        if fid < nwords:
                            if nw_line < MAX_TOK:
                                line_wids[nw_line] = fid
                                line_whash[nw_line] = fnv
                                nw_line += 1
                        else:
                            if nl_line < MAX_TOK:
                                line_lids[nl_line] = fid
                                nl_line += 1
                    in_token = False
                pos += 1
                continue

            if not in_token:
                tok_start = pos
                in_token = True
            pos += 1

        # ── process complete line (modes 0-4) ──
        n_ctx = np.int32(0)
        n_targets = np.int32(0)
        do_train = False

        if mode == 0:
            if nw_line > 0 and nl_line > 0:
                n_ctx = _build_word_ctx(
                    line_wids, line_whash, nw_line,
                    word_ngrams, bucket, ngram_base, ctx_buf)
                rng_state = (rng_state * np.int64(48271)) % \
                    np.int64(2147483647)
                target_buf[0] = line_lids[np.int32(
                    np.uint64(rng_state) % np.uint64(nl_line))]
                n_targets = np.int32(1)
                do_train = True

        elif mode == 1:
            if nl_line >= 2:
                rng_state = (rng_state * np.int64(48271)) % \
                    np.int64(2147483647)
                idx = np.int32(
                    np.uint64(rng_state) % np.uint64(nl_line))
                n_ctx = _build_word_ctx(
                    line_wids, line_whash, nw_line,
                    word_ngrams, bucket, ngram_base, ctx_buf)
                for k in range(nl_line):
                    if k != idx:
                        ctx_buf[n_ctx] = line_lids[k]
                        n_ctx += 1
                target_buf[0] = line_lids[idx]
                n_targets = np.int32(1)
                do_train = True

        elif mode == 2:
            if nl_line >= 2:
                rng_state = (rng_state * np.int64(48271)) % \
                    np.int64(2147483647)
                idx = np.int32(
                    np.uint64(rng_state) % np.uint64(nl_line))
                n_ctx = _build_word_ctx(
                    line_wids, line_whash, nw_line,
                    word_ngrams, bucket, ngram_base, ctx_buf)
                ctx_buf[n_ctx] = line_lids[idx]
                n_ctx += 1
                nt = np.int32(0)
                for k in range(nl_line):
                    if k != idx:
                        target_buf[nt] = line_lids[k]
                        nt += 1
                n_targets = nt
                do_train = True

        elif mode == 3:
            if nl_line >= 2:
                rng_state = (rng_state * np.int64(48271)) % \
                    np.int64(2147483647)
                i1 = np.int32(
                    np.uint64(rng_state) % np.uint64(nl_line))
                rng_state = (rng_state * np.int64(48271)) % \
                    np.int64(2147483647)
                i2 = np.int32(
                    np.uint64(rng_state) % np.uint64(nl_line))
                while i2 == i1:
                    rng_state = (rng_state * np.int64(48271)) \
                        % np.int64(2147483647)
                    i2 = np.int32(
                        np.uint64(rng_state) % np.uint64(
                            nl_line))
                n_ctx = _build_word_ctx(
                    line_wids, line_whash, nw_line,
                    word_ngrams, bucket, ngram_base, ctx_buf)
                ctx_buf[n_ctx] = line_lids[i1]
                n_ctx += 1
                target_buf[0] = line_lids[i2]
                n_targets = np.int32(1)
                do_train = True

        elif mode == 4:
            if nl_line >= 2:
                n_ctx = _build_word_ctx(
                    line_wids, line_whash, nw_line,
                    word_ngrams, bucket, ngram_base, ctx_buf)
                ctx_buf[n_ctx] = line_lids[0]
                n_ctx += 1
                target_buf[0] = line_lids[1]
                n_targets = np.int32(1)
                do_train = True

        if do_train:
            if cur_lr <= np.float32(0.0):
                break
            # buffer into batch
            bi = n_buffered
            for ci2 in range(n_ctx):
                batch_ctx[bi, ci2] = ctx_buf[ci2]
            batch_n_ctx[bi] = n_ctx
            for ti in range(n_targets):
                batch_tgt[bi, ti] = target_buf[ti]
            batch_n_tgt[bi] = n_targets
            n_buffered += 1

            if n_buffered >= BS:
                loss, rng_state = _train_batch(
                    emb, adagrad_lhs, adagrad_rhs,
                    batch_ctx, batch_n_ctx,
                    batch_tgt, batch_n_tgt,
                    n_buffered,
                    batch_lhs, batch_rhs_pos, rhs_neg,
                    batch_neg_mean, batch_grad_w,
                    neg_ids_b, batch_neg_flags,
                    batch_n_valid,
                    dim, nwords, nlabels, margin,
                    neg_search_limit,
                    cur_lr, rng_state, norm_limit, mode,
                    max_neg_samples, neg_pool, neg_pool_size,
                    neg_vecs,
                    ex_labels, ex_offsets, n_multi_ex,
                    neg_label_buf, neg_label_cnt)
                loss_sum += loss
                n_steps += n_buffered
                n_buffered = np.int32(0)

            # stepwise LR decay every 1000 samples
            sample_count += 1
            if sample_count % K_DECR_STEP == 0:
                cur_lr -= decr_per_k

    # flush remaining buffered examples
    if n_buffered > 0:
        loss, rng_state = _train_batch(
            emb, adagrad_lhs, adagrad_rhs,
            batch_ctx, batch_n_ctx,
            batch_tgt, batch_n_tgt,
            n_buffered,
            batch_lhs, batch_rhs_pos, rhs_neg,
            batch_neg_mean, batch_grad_w,
            neg_ids_b, batch_neg_flags,
            batch_n_valid,
            dim, nwords, nlabels, margin,
            neg_search_limit,
            cur_lr, rng_state, norm_limit, mode,
            max_neg_samples, neg_pool, neg_pool_size,
            neg_vecs,
            ex_labels, ex_offsets, n_multi_ex,
            neg_label_buf, neg_label_cnt)
        loss_sum += loss
        n_steps += n_buffered

    return loss_sum, n_steps, rng_state


# ── vocabulary ───────────────────────────────────────────────────────────────


@dataclass
class Vocab:
    words: list[str]            = field(default_factory=list)
    labels: list[str]           = field(default_factory=list)
    w2i: dict[str, int]         = field(default_factory=dict)
    l2i: dict[str, int]         = field(default_factory=dict)
    whash: dict[str, int]       = field(default_factory=dict)
    ntokens: int                = 0
    bucket: int                 = 0
    word_ngrams: int            = 1
    label_prefix: str           = "__label__"

    @property
    def nwords(self) -> int:
        return len(self.words)

    @property
    def nlabels(self) -> int:
        return len(self.labels)

    def tokenise_line(self, tokens: list[str]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse tokens → (word_ids, word_hashes, label_emb_ids)."""
        w2i, whash, l2i = self.w2i, self.whash, self.l2i
        prefix = self.label_prefix
        nw = self.nwords

        word_ids, word_hashes, label_ids = [], [], []
        for tok in tokens:
            if tok.startswith(prefix):
                lid = l2i.get(tok)
                if lid is not None:
                    label_ids.append(nw + lid)
            else:
                wid = w2i.get(tok, -1)
                if wid >= 0:
                    word_ids.append(wid)
                    word_hashes.append(whash[tok])

        return (np.array(word_ids, dtype=np.int32),
                np.array(word_hashes, dtype=np.int32),
                np.array(label_ids, dtype=np.int32))


# ── hash table helpers (Pass 2) ──────────────────────────────────────────────


def _alloc_hash_table(estimated_unique):
    """Allocate struct-of-arrays hash table (power-of-2 size)."""
    table_size = 1
    while table_size < max(estimated_unique * 4, 1 << 16):
        table_size <<= 1
    mask = np.int64(table_size - 1)
    return (np.zeros(table_size, np.int32),   # ht_fnv
            np.zeros(table_size, np.int8),     # ht_occ
            np.zeros(table_size, np.int64),    # ht_tok_start
            np.zeros(table_size, np.int32),    # ht_tok_len
            np.zeros(table_size, np.int64),    # ht_freq
            np.zeros(table_size, np.int8),     # ht_is_label
            mask, table_size)


def _extract_vocab(buf, ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                   ht_freq, ht_is_label, n_tokens, *,
                   min_count, bucket, word_ngrams, label_prefix,
                   verbose):
    """Pass 2: extract Vocab + remap from hash table."""
    table_size = len(ht_occ)
    occupied = np.where(ht_occ == 1)[0]

    word_slots = occupied[ht_is_label[occupied] == 0]
    label_slots = occupied[ht_is_label[occupied] == 1]

    word_slots = word_slots[ht_freq[word_slots] >= min_count]

    word_slots = word_slots[np.argsort(-ht_freq[word_slots])]
    label_slots = label_slots[np.argsort(-ht_freq[label_slots])]

    nwords = len(word_slots)
    nlabels = len(label_slots)

    remap = np.full(table_size, -1, dtype=np.int32)
    for i, slot in enumerate(word_slots):
        remap[slot] = i
    for i, slot in enumerate(label_slots):
        remap[slot] = nwords + i

    words, whash = [], {}
    for i, slot in enumerate(word_slots):
        s = int(ht_tok_start[slot])
        e = s + int(ht_tok_len[slot])
        w = bytes(buf[s:e]).decode("utf-8", errors="replace")
        words.append(w)
        whash[w] = int(ht_fnv[slot])
    labels = []
    for slot in label_slots:
        s = int(ht_tok_start[slot])
        e = s + int(ht_tok_len[slot])
        labels.append(bytes(buf[s:e]).decode("utf-8", errors="replace"))

    w2i = {w: i for i, w in enumerate(words)}
    l2i = {l: i for i, l in enumerate(labels)}
    bkt = bucket if word_ngrams > 1 else 0

    if verbose > 0:
        print(f"\rRead {int(n_tokens) // 1_000_000}M words — "
              f"vocab {nwords} words, {nlabels} labels "
              f"(min_count={min_count})", file=sys.stderr)

    vocab = Vocab(words=words, labels=labels, w2i=w2i, l2i=l2i,
                  whash=whash, ntokens=int(n_tokens), bucket=bkt,
                  word_ngrams=word_ngrams, label_prefix=label_prefix)
    return vocab, remap


# ── model ────────────────────────────────────────────────────────────────────


class StarSpace:
    """Pure-Python StarSpace classifier/embedder (trainMode 0-4).

    ::

        model = StarSpace.train("train.txt", dim=100, epoch=5)
        model.predict("the food was great")
    """

    __slots__ = ("emb", "vocab", "dim", "word_ngrams", "margin",
                 "neg_search_limit", "lr", "epoch", "seed",
                 "norm_limit", "verbose", "train_mode",
                 "max_neg_samples", "batch_size")

    def __init__(self, *, vocab: Vocab, emb: np.ndarray,
                 dim: int, word_ngrams: int = 1, margin: float = 0.05,
                 neg_search_limit: int = 50, lr: float = 0.01,
                 epoch: int = 5, seed: int = 0, norm_limit: float = 1.0,
                 verbose: int = 2, train_mode: int = 0,
                 max_neg_samples: int = 10, batch_size: int = 5):
        self.vocab, self.emb = vocab, emb
        self.dim = dim
        self.word_ngrams = word_ngrams
        self.margin = margin
        self.neg_search_limit = neg_search_limit
        self.lr, self.epoch, self.seed = lr, epoch, seed
        self.norm_limit = norm_limit
        self.verbose = verbose
        self.train_mode = train_mode
        self.max_neg_samples = max_neg_samples
        self.batch_size = batch_size

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, text: str, k: int = 1) -> list[tuple[str, float]]:
        """Predict top-k labels. Returns [(label, cosine_similarity), ...]."""
        v = self.vocab
        if v.nlabels == 0:
            return []
        word_ids, word_hashes, label_ids = v.tokenise_line(text.split())
        if len(word_ids) == 0 and len(label_ids) == 0:
            return []

        input_ids = list(word_ids)
        if v.word_ngrams > 1 and v.bucket > 0:
            _M = np.uint64(0xFFFFFFFFFFFFFFFF)
            ngram_base = v.nwords + v.nlabels
            for i in range(len(word_hashes)):
                hv = np.uint64(np.int64(word_hashes[i])) & _M
                for j in range(i + 1, min(len(word_hashes),
                                          i + v.word_ngrams)):
                    hv = (hv * np.uint64(116049371) +
                          (np.uint64(np.int64(word_hashes[j])) & _M)) & _M
                    input_ids.append(ngram_base + int(hv % np.uint64(v.bucket)))
        # Include label embeddings in LHS (for modes 1-4 where query
        # may contain labels, e.g. recommendation / collaborative filtering)
        input_ids.extend(label_ids)

        lhs = np.zeros(self.dim, np.float32)
        for idx in input_ids:
            lhs += self.emb[idx]
        n = np.linalg.norm(lhs)
        if n > 0:
            lhs /= n

        nw = v.nwords
        label_embs = self.emb[nw:nw + v.nlabels]
        norms = np.linalg.norm(label_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        sims = (label_embs / norms) @ lhs

        top_k = np.argsort(sims)[::-1][:k]
        return [(v.labels[i], float(sims[i])) for i in top_k]

    def test(self, data, k: int = 1) -> tuple[int, float, float]:
        """Evaluate on labeled data. Returns (N, precision@k, recall@k)."""
        if isinstance(data, str):
            data = iter_lines(data)
        v = self.vocab
        n = 0
        p_sum = 0.0
        r_sum = 0.0

        for tokens in data:
            true_labels = set()
            text_tokens = []
            for tok in tokens:
                if tok.startswith(v.label_prefix):
                    true_labels.add(tok)
                else:
                    text_tokens.append(tok)

            if not true_labels or not text_tokens:
                continue

            preds = self.predict(" ".join(text_tokens), k=k)
            pred_labels = {label for label, _ in preds}

            matches = len(pred_labels & true_labels)
            p_sum += matches / max(len(pred_labels), 1)
            r_sum += matches / len(true_labels)
            n += 1

        precision = p_sum / max(n, 1)
        recall = r_sum / max(n, 1)
        return n, precision, recall

    # ── I/O ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez_compressed(
            path,
            emb=self.emb,
            words=np.array(self.vocab.words, dtype=object),
            labels=np.array(self.vocab.labels, dtype=object),
            meta=np.array([self.dim, self.word_ngrams, self.epoch, self.seed,
                           self.vocab.ntokens, self.vocab.bucket,
                           self.neg_search_limit, self.train_mode,
                           self.max_neg_samples, self.batch_size]),
            fmeta=np.array([self.lr, self.margin, self.norm_limit]),
            label_prefix=np.array([self.vocab.label_prefix]),
        )

    @classmethod
    def load(cls, path: str) -> StarSpace:
        d = np.load(path, allow_pickle=True)
        words = list(d["words"])
        labels = list(d["labels"])
        m = d["meta"]
        fm = d["fmeta"]
        lp = str(d["label_prefix"][0])

        whash = {w: int(_fnv1a_bytes(
            np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
            for w in words}
        vocab = Vocab(
            words=words, labels=labels,
            w2i={w: i for i, w in enumerate(words)},
            l2i={l: i for i, l in enumerate(labels)},
            whash=whash,
            ntokens=int(m[4]),
            bucket=int(m[5]),
            word_ngrams=int(m[1]),
            label_prefix=lp,
        )
        train_mode = int(m[7]) if len(m) > 7 else 0
        max_neg_samples = int(m[8]) if len(m) > 8 else 10
        batch_size_val = int(m[9]) if len(m) > 9 else 5
        return cls(
            vocab=vocab, emb=d["emb"],
            dim=int(m[0]), word_ngrams=int(m[1]),
            epoch=int(m[2]), seed=int(m[3]),
            neg_search_limit=int(m[6]),
            lr=float(fm[0]), margin=float(fm[1]),
            norm_limit=float(fm[2]), verbose=2,
            train_mode=train_mode,
            max_neg_samples=max_neg_samples,
            batch_size=batch_size_val,
        )

    # ── training (mmap pipeline, no intermediate arrays) ─────────────────

    @classmethod
    def train(cls, data, *, dim=100, epoch=5, lr=0.01,
              margin=0.05, neg_search_limit=50, min_count=1,
              word_ngrams=1, bucket=2_000_000, norm_limit=1.0,
              init_rand_sd=0.001, seed=0, verbose=2,
              train_mode=0, max_neg_samples=10,
              batch_size=5, label="__label__") -> StarSpace:
        """Train a StarSpace model.

        *data* is a file path (str) or an iterable of token lists.
        File paths use the Numba mmap pipeline (no intermediate arrays).
        Iterables are spilled to a temp file first.
        """
        if not isinstance(data, str):
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8")
            try:
                for tokens in data:
                    tmp.write(" ".join(tokens) + "\n")
                tmp.close()
                return cls.train(
                    tmp.name, dim=dim, epoch=epoch, lr=lr, margin=margin,
                    neg_search_limit=neg_search_limit, min_count=min_count,
                    word_ngrams=word_ngrams, bucket=bucket,
                    norm_limit=norm_limit, init_rand_sd=init_rand_sd,
                    seed=seed, verbose=verbose, train_mode=train_mode,
                    max_neg_samples=max_neg_samples,
                    batch_size=batch_size, label=label)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        path = data

        # mmap the text file
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        buf_len = np.int64(len(buf))

        # Pass 1: vocab scan
        label_prefix_bytes = np.frombuffer(
            label.encode("utf-8"), dtype=np.uint8)

        estimated = min(int(buf_len) // 4, 4_000_000)
        (ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
         ht_freq, ht_is_label, mask, table_size) = _alloc_hash_table(
             estimated)

        n_unique, n_tokens, n_lines, status = _vocab_scan(
            buf, buf_len, ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
            ht_freq, ht_is_label, mask, label_prefix_bytes)

        while status == -1:
            estimated = int(n_unique) * 4
            (ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
             ht_freq, ht_is_label, mask, table_size) = _alloc_hash_table(
                 estimated)
            n_unique, n_tokens, n_lines, status = _vocab_scan(
                buf, buf_len, ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                ht_freq, ht_is_label, mask, label_prefix_bytes)

        # Pass 2: extract vocab
        vocab, remap = _extract_vocab(
            buf, ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
            ht_freq, ht_is_label, n_tokens,
            min_count=min_count, bucket=bucket, word_ngrams=word_ngrams,
            label_prefix=label, verbose=verbose)

        # Build frequency-weighted negative sampling pool
        # (matches native C++ getRandomRHS / getRandomWord)
        pool_rng = np.random.RandomState(seed + 42)
        # Count trainable examples: C++ numSamples = data->getSize()
        # which is the filtered count after parser.check(). Used for LR decay.
        n_train_ex = 0
        if train_mode in (1, 2, 3, 4):
            # modes 1-4: C++ filters single-label examples at load time,
            # so getRandomRHS only draws from multi-label examples.
            # Build neg pool from label frequencies of 2+ label examples only.
            label_set = set(vocab.labels)
            nw = vocab.nwords
            ml_freq: dict[int, float] = {}
            for tokens in iter_lines(path):
                lids = [nw + vocab.l2i[t]
                        for t in tokens if t in label_set]
                if len(lids) >= 2:
                    n_train_ex += 1
                    # C++ getRandomRHS: uniform random example, then
                    # uniform random label → weight = 1/|labels_in_ex|
                    w = 1.0 / len(lids)
                    for lid in lids:
                        ml_freq[lid] = ml_freq.get(lid, 0.0) + w
            if ml_freq:
                ids = np.array(list(ml_freq.keys()), dtype=np.int32)
                freqs = np.array(list(ml_freq.values()), dtype=np.float64)
                probs = freqs / freqs.sum()
                pool_size = min(10_000_000, max(100_000, len(ids) * 100))
                neg_pool = pool_rng.choice(
                    ids, size=pool_size, p=probs).astype(np.int32)
            else:
                neg_pool = np.zeros(0, dtype=np.int32)
        else:
            # mode 0: C++ getRandomRHS picks uniform random example,
            # then uniform random label → weight = 1/|labels_in_ex|.
            # Mode 0 keeps ALL examples (including single-label).
            label_set = set(vocab.labels)
            word_set = set(vocab.words)
            nw = vocab.nwords
            m0_freq: dict[int, float] = {}
            for tokens in iter_lines(path):
                lids = [nw + vocab.l2i[t]
                        for t in tokens if t in label_set]
                has_words = any(t in word_set for t in tokens)
                if len(lids) >= 1 and has_words:
                    n_train_ex += 1
                    w = 1.0 / len(lids)
                    for lid in lids:
                        m0_freq[lid] = m0_freq.get(lid, 0.0) + w
            if m0_freq:
                ids = np.array(list(m0_freq.keys()), dtype=np.int32)
                freqs = np.array(list(m0_freq.values()), dtype=np.float64)
                probs = freqs / freqs.sum()
                pool_size = min(10_000_000, max(100_000, len(ids) * 100))
                neg_pool = pool_rng.choice(
                    ids, size=pool_size, p=probs).astype(np.int32)
            else:
                neg_pool = np.zeros(0, dtype=np.int32)

        del ht_freq, ht_is_label

        # init model
        n_emb = vocab.nwords + vocab.nlabels + vocab.bucket
        rng = np.random.RandomState(seed)
        emb = rng.normal(0, init_rand_sd, (n_emb, dim)).astype(np.float32)

        model = cls(vocab=vocab, emb=emb, dim=dim,
                    word_ngrams=word_ngrams, margin=margin,
                    neg_search_limit=neg_search_limit,
                    lr=lr, epoch=epoch, seed=seed, norm_limit=norm_limit,
                    verbose=verbose, train_mode=train_mode,
                    max_neg_samples=max_neg_samples,
                    batch_size=batch_size)

        # train (retokenises from mmap each epoch)
        model._fit(path, buf, buf_len, ht_fnv, ht_occ, ht_tok_start,
                   ht_tok_len, mask, remap, neg_pool, n_train_ex)
        return model

    def _fit(self, path, buf, buf_len, ht_fnv, ht_occ, ht_tok_start,
             ht_tok_len, mask, remap, neg_pool, n_train_ex):
        v = self.vocab
        rng_state = np.int64(self.seed + 1)
        # separate LHS/RHS AdaGrad accumulators (matches native LHSUpdates_ / RHSUpdates_)
        adagrad_lhs = np.zeros(self.emb.shape[0], np.float32)
        adagrad_rhs = np.zeros(self.emb.shape[0], np.float32)
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()

        neg_pool_size = np.int32(len(neg_pool))

        # mode 2: build per-example label index for multi-label negatives
        # (matches native getRandomRHS which returns all labels from a
        #  random example minus one)
        if self.train_mode == 2:
            label_set = set(v.labels)
            label_to_id = {l: v.nwords + i for i, l in enumerate(v.labels)}
            _ex_list, _off_list = [], [0]
            for tokens in iter_lines(path):
                lids = [label_to_id[t] for t in tokens if t in label_set]
                if len(lids) >= 2:
                    _ex_list.extend(lids)
                    _off_list.append(len(_ex_list))
            ex_labels = np.array(_ex_list, dtype=np.int32)
            ex_offsets = np.array(_off_list, dtype=np.int64)
        else:
            ex_labels = np.empty(0, dtype=np.int32)
            ex_offsets = np.zeros(1, dtype=np.int64)
        n_multi_ex = np.int32(len(ex_offsets) - 1)

        # pre-compute line offsets for per-epoch shuffling
        line_offsets = _find_line_offsets(buf, buf_len)
        n_lines = np.int64(len(line_offsets))
        perm = np.arange(n_lines, dtype=np.int64)

        # epoch-based LR schedule (matches native C++ StarSpace::train)
        # rate decays from lr to ~1e-9 over all epochs
        decr_per_epoch = (self.lr - 1e-9) / max(self.epoch, 1)
        epoch_rate = self.lr

        for ep in range(self.epoch):
            finish_rate = max(epoch_rate - decr_per_epoch, 0.0)

            loss, steps, rng_state = _train_epoch(
                self.emb, adagrad_lhs, adagrad_rhs, buf, buf_len,
                ht_fnv, ht_occ, ht_tok_start, ht_tok_len,
                mask, remap,
                np.int32(v.nwords), np.int32(v.nlabels), np.int32(self.dim),
                np.int32(self.word_ngrams), np.int32(v.bucket),
                np.float32(self.margin), np.int32(self.neg_search_limit),
                np.float32(epoch_rate), np.float32(max(finish_rate, 0.0)),
                rng_state, np.float32(self.norm_limit),
                np.int32(self.train_mode),
                line_offsets, n_lines, perm,
                np.int32(self.max_neg_samples), neg_pool, neg_pool_size,
                np.int32(self.batch_size),
                ex_labels, ex_offsets, n_multi_ex,
                np.int64(n_train_ex))

            loss_acc += float(loss)
            n_acc += int(steps)
            epoch_rate -= decr_per_epoch

            if self.verbose > 0:
                elapsed = max(time.time() - t0, 1e-6)
                pct = (ep + 1) / self.epoch * 100.0
                avg = loss_acc / max(n_acc, 1) / self.neg_search_limit
                print(f"\r{pct:5.1f}%  pass={ep + 1}/{self.epoch}"
                      f"  loss={avg:.4f}  ({elapsed:.1f}s)",
                      end="", file=sys.stderr)

        if self.verbose > 0:
            avg = loss_acc / max(n_acc, 1) / self.neg_search_limit
            print(f"\rDone — avg loss {avg:.4f}"
                  f"  ({time.time() - t0:.1f}s)", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli():
    p = argparse.ArgumentParser(prog="starspace")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("corpus")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--dim",              type=int,   default=100)
    tr.add_argument("--epoch",            type=int,   default=5)
    tr.add_argument("--lr",               type=float, default=0.01)
    tr.add_argument("--margin",           type=float, default=0.05)
    tr.add_argument("--neg-search-limit", type=int,   default=50)
    tr.add_argument("--min-count",        type=int,   default=1)
    tr.add_argument("--word-ngrams",      type=int,   default=1)
    tr.add_argument("--bucket",           type=int,   default=2_000_000)
    tr.add_argument("--norm-limit",       type=float, default=1.0)
    tr.add_argument("--init-rand-sd",     type=float, default=0.001)
    tr.add_argument("--seed",             type=int,   default=0)
    tr.add_argument("--train-mode",       type=int,   default=0)
    tr.add_argument("--max-neg-samples",  type=int,   default=10)
    tr.add_argument("--batch-size",       type=int,   default=5)

    ts = sub.add_parser("test")
    ts.add_argument("model")
    ts.add_argument("test_file")
    ts.add_argument("-k", type=int, default=1)

    pr = sub.add_parser("predict")
    pr.add_argument("model")
    pr.add_argument("-k", type=int, default=1)

    args = p.parse_args()
    if args.cmd == "train":
        m = StarSpace.train(
            args.corpus,
            dim=args.dim, epoch=args.epoch, lr=args.lr,
            margin=args.margin, neg_search_limit=args.neg_search_limit,
            min_count=args.min_count, word_ngrams=args.word_ngrams,
            bucket=args.bucket, norm_limit=args.norm_limit,
            init_rand_sd=args.init_rand_sd, seed=args.seed,
            train_mode=args.train_mode,
            max_neg_samples=args.max_neg_samples,
            batch_size=args.batch_size)
        m.save(args.output)
    elif args.cmd == "test":
        m = StarSpace.load(args.model)
        n, prec, rec = m.test(args.test_file, k=args.k)
        print(f"N\t{n}")
        print(f"P@{args.k}\t{prec:.4f}")
        print(f"R@{args.k}\t{rec:.4f}")
    elif args.cmd == "predict":
        m = StarSpace.load(args.model)
        for line in sys.stdin:
            preds = m.predict(line.strip(), k=args.k)
            print(" ".join(f"{label} {sim:.4f}" for label, sim in preds))
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
