"""Paper recommendation — collaborative filtering on research papers.

Trains StarSpace (mode 1) on paper co-occurrence within user collections.
Papers grouped together by researchers in their library collections are
treated as related — like users and artists in Last.FM.

Each training line: title words + author names + journal + year
+ P-prefixed IDs of co-occurring papers. The model learns content-based
(metadata → papers) and collaborative (papers → papers) recommendation.

Dataset: ~/work/data/papers/ (library.jsonl + projects.jsonl)
"""

import json
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from starspace import StarSpace

DATA_DIR = Path.home() / "work" / "data" / "papers"
LIBRARY_FILE = str(DATA_DIR / "library.jsonl")
PROJECTS_FILE = str(DATA_DIR / "projects.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(LIBRARY_FILE),
    reason="Papers dataset not found (need ~/work/data/papers/)")

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been", "its",
    "it", "this", "that", "as", "but", "not", "no", "has", "have", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "shall", "their", "they", "them", "we", "our", "us", "you",
    "your", "he", "she", "him", "her", "his", "who", "which", "what",
    "where", "when", "how", "than", "then", "so", "if", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "into", "over", "after", "before", "between", "under", "about",
    "up", "out", "one", "two", "three", "also", "new", "use", "used",
    "using", "based", "via", "through", "during", "among", "upon",
    "study", "analysis", "approach", "method", "methods", "results",
    "effect", "effects", "role", "case", "review", "model", "models",
    "evidence", "impact", "implications",
}


def _clean_title_words(title):
    """Extract meaningful words from a paper title."""
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s-]', ' ', title)
    return [w for w in title.split() if len(w) >= 3 and w not in STOPWORDS]


def _author_token(name):
    """Author name → 'AU_firstname_lastname' token (spaces → underscores)."""
    tok = re.sub(r'[^a-z0-9]', '_', name.lower().strip())
    tok = re.sub(r'_+', '_', tok).strip('_')
    return "AU_" + tok if len(tok) >= 2 else ""


def _journal_tokens(journal):
    """Journal name → 'J_word' tokens."""
    journal = journal.lower()
    journal = re.sub(r'[^a-z0-9\s]', ' ', journal)
    return ["J_" + w for w in journal.split()
            if len(w) >= 3 and w not in STOPWORDS]


def _paper_features(cid, paper_titles, paper_authors, paper_journals):
    """Build all word features for a paper: title + authors + journal."""
    features = _clean_title_words(paper_titles.get(cid, ""))
    for author in paper_authors.get(cid, []):
        tok = _author_token(author)
        if tok:
            features.append(tok)
    features.extend(_journal_tokens(paper_journals.get(cid, "")))
    return features


_data_cache: dict = {}


def _make_group_lines(group_cids, paper_titles, paper_authors,
                      paper_journals, max_neighbors, rng):
    """Create training lines for one paper group.

    One line per paper: metadata features + P-prefixed IDs of co-occurring papers.
    """
    lines = []
    for i, cid in enumerate(group_cids):
        features = _paper_features(cid, paper_titles, paper_authors,
                                   paper_journals)
        if len(features) < 2:
            continue
        neighbors = [c for j, c in enumerate(group_cids) if j != i]
        if len(neighbors) > max_neighbors:
            neighbors = rng.sample(neighbors, max_neighbors)
        labels = ["P" + c for c in neighbors]
        if not labels:
            continue
        lines.append(" ".join(features + labels))
    return lines


def _get_paper_data():
    """Build and cache training data from paper groups.

    Two sources of paper groups:
    1. Library (user, collection) groups — papers a user filed together
    2. Project source groups — papers a researcher collected for a project

    Holds out 20% of projects for recommendation evaluation.
    Returns (train_path, holdout_groups, holdout_projects,
             paper_titles, lib_to_corpus).
    """
    if _data_cache:
        return (_data_cache["path"], _data_cache["holdout_groups"],
                _data_cache["projects"],
                _data_cache["titles"], _data_cache["lib_to_corpus"])

    rng = random.Random(42)

    # Read library → groups + metadata
    coll_groups: dict[tuple, list] = defaultdict(list)
    user_papers: dict[str, list] = defaultdict(list)
    paper_titles: dict[str, str] = {}
    paper_authors: dict[str, list] = {}
    paper_journals: dict[str, str] = {}
    lib_to_corpus: dict[str, str] = {}

    with open(LIBRARY_FILE) as f:
        for raw in f:
            e = json.loads(raw)
            eid = e.get("id", "")
            cid = e.get("corpusid", "")
            title = e.get("title", "")
            user = e.get("user", "")
            if not cid or not title or not user:
                continue
            paper_titles[cid] = title
            if e.get("authors"):
                paper_authors[cid] = e["authors"]
            if e.get("journal"):
                paper_journals[cid] = e["journal"]
            if eid:
                lib_to_corpus[eid] = cid
            user_papers[user].append(cid)
            for c in e.get("collections", []):
                coll_groups[(user, c)].append(cid)

    # Filter collection groups: 3-50 papers, deduplicate
    filtered_coll = []
    for cids in coll_groups.values():
        cids = list(dict.fromkeys(cids))
        if 3 <= len(cids) <= 50:
            filtered_coll.append(cids)

    # User-level library groups: all papers per user (3-30)
    user_groups = []
    for cids in user_papers.values():
        cids = list(dict.fromkeys(cids))
        if 3 <= len(cids) <= 30:
            user_groups.append(cids)

    # Split collection groups 90/10 for holdout
    rng.shuffle(filtered_coll)
    n_holdout = max(len(filtered_coll) // 10, 1)
    holdout_groups = filtered_coll[:n_holdout]
    train_coll_groups = filtered_coll[n_holdout:]

    # Load project groups — resolve both sources and collections
    project_groups = []
    with open(PROJECTS_FILE) as f:
        for raw in f:
            p = json.loads(raw)
            title = p.get("title", "")
            if not title:
                continue
            owner = p.get("owner", "")
            # Sources → corpus IDs
            cids = set()
            for s in p.get("sources", []):
                c = lib_to_corpus.get(s, "")
                if c:
                    cids.add(c)
            # Collections → corpus IDs via (owner, collection_name)
            for cname in p.get("collections", []):
                for c in coll_groups.get((owner, cname), []):
                    cids.add(c)
            cids = list(dict.fromkeys(cids))
            if len(cids) >= 3:
                project_groups.append((title, cids))

    # Build training lines from all sources
    max_neighbors = 15
    meta = (paper_titles, paper_authors, paper_journals)
    lines = []
    for cids in train_coll_groups:
        lines.extend(_make_group_lines(cids, *meta, max_neighbors, rng))
    for _, cids in project_groups:
        lines.extend(_make_group_lines(cids, *meta, max_neighbors, rng))
    for cids in user_groups:
        lines.extend(_make_group_lines(cids, *meta, max_neighbors, rng))

    # Single-paper lines for ALL library papers — ensures every paper
    # has a trained embedding and is recommendable via content similarity
    for cid in paper_titles:
        features = _paper_features(cid, paper_titles, paper_authors,
                                   paper_journals)
        if len(features) >= 2:
            lines.append(" ".join(features + ["P" + cid]))

    rng.shuffle(lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ss_papers_")
    for line in lines:
        tmp.write(line + "\n")
    tmp.close()

    _data_cache.update({
        "path": tmp.name, "holdout_groups": holdout_groups,
        "projects": project_groups,
        "titles": paper_titles, "lib_to_corpus": lib_to_corpus,
    })
    return (tmp.name, holdout_groups, project_groups,
            paper_titles, lib_to_corpus)


@pytest.fixture(scope="module")
def paper_model():
    """Train on paper co-occurrence data, mode 1."""
    path, _, _, _, _ = _get_paper_data()
    return StarSpace.train(
        path, dim=100, epoch=40, lr=0.01,
        train_mode=1, bucket=100000, verbose=0,
        neg_search_limit=100, max_neg_samples=50,
        label="P", min_count=1)


@pytest.fixture(scope="module")
def paper_titles():
    """Corpus ID → paper title mapping."""
    _, _, _, titles, _ = _get_paper_data()
    return titles


@pytest.fixture(scope="module")
def holdout_groups():
    """Holdout collection groups for evaluation."""
    _, holdout, _, _, _ = _get_paper_data()
    return holdout


@pytest.fixture(scope="module")
def all_projects():
    """All projects with 3+ resolved sources."""
    _, _, projects, _, _ = _get_paper_data()
    return projects


@pytest.fixture(scope="module", autouse=True)
def _cleanup():
    """Clean up temp training file after all tests."""
    yield
    path = _data_cache.get("path")
    if path and os.path.isfile(path):
        os.unlink(path)


class TestPaperRecommendation:
    def test_learns_vocab(self, paper_model):
        """Should learn title words and paper ID labels."""
        nw = paper_model.vocab.nwords
        nl = paper_model.vocab.nlabels
        print(f"\n  Words: {nw}, Paper IDs: {nl}")
        assert nw > 1000
        assert nl > 1000

    def test_collaborative_filtering(self, paper_model, paper_titles,
                                     holdout_groups):
        """Given paper IDs from a group, predict other group members."""
        rng = random.Random(42)
        test_groups = [g for g in holdout_groups if len(g) >= 5]
        if len(test_groups) > 30:
            test_groups = rng.sample(test_groups, 30)

        hits = 0
        k = 20
        print(f"\n  Collaborative filtering — holdout groups (top {k}):")
        print(f"  {'':5s} {'query papers':50s}  {'found'}")
        print(f"  {'':5s} {'-'*50}  {'-'*50}")
        for group in test_groups:
            mid = len(group) // 2
            query_cids = group[:mid]
            expect_cids = set(group[mid:])

            query = " ".join("P" + c for c in query_cids)
            preds = paper_model.predict(query, k=k)
            if not preds:
                continue
            pred_cids = {lbl[1:] for lbl, _ in preds}
            found = pred_cids & expect_cids
            if found:
                hits += 1
                # Show first few hits
                if hits <= 5:
                    q_str = paper_titles.get(query_cids[0], "?")[:50]
                    f_str = paper_titles.get(list(found)[0], "?")[:50]
                    print(f"  OK    {q_str:50s}  {f_str}")

        pct = hits / len(test_groups) if test_groups else 0
        print(f"\n  {hits}/{len(test_groups)} groups: held-out papers found"
              f" ({pct:.0%})")

    def test_content_recommendation(self, paper_model, paper_titles,
                                    holdout_groups):
        """Paper title words find co-occurring papers (content-based)."""
        rng = random.Random(42)
        test_groups = [g for g in holdout_groups if len(g) >= 5]
        if len(test_groups) > 30:
            test_groups = rng.sample(test_groups, 30)

        hits = 0
        k = 20
        print(f"\n  Content-based — title words → papers (top {k}):")
        print(f"  {'':5s} {'query title':50s}  {'found paper'}")
        print(f"  {'':5s} {'-'*50}  {'-'*50}")
        for group in test_groups:
            target_cid = group[0]
            expect_cids = set(group[1:])

            title = paper_titles.get(target_cid, "")
            words = _clean_title_words(title)
            if len(words) < 3:
                continue

            preds = paper_model.predict(" ".join(words), k=k)
            if not preds:
                continue
            pred_cids = {lbl[1:] for lbl, _ in preds}
            found = pred_cids & expect_cids
            if found:
                hits += 1
                if hits <= 5:
                    t_str = title[:50]
                    f_str = paper_titles.get(list(found)[0], "?")[:50]
                    print(f"  OK    {t_str:50s}  {f_str}")

        pct = hits / len(test_groups) if test_groups else 0
        print(f"\n  {hits}/{len(test_groups)} groups: co-occurring papers"
              f" found by title ({pct:.0%})")

    def test_project_leave_one_out(self, paper_model, all_projects,
                                   paper_titles):
        """Leave-one-out: predict held-out source paper from remaining."""
        rng = random.Random(42)
        test_projects = [
            (t, c) for t, c in all_projects if len(c) >= 5]
        if len(test_projects) > 30:
            test_projects = rng.sample(test_projects, 30)

        hits = 0
        k = 20
        print(f"\n  Project leave-one-out — holdout projects (top {k}):")
        print(f"  {'':5s} {'project':35s}  {'held-out paper'}")
        print(f"  {'':5s} {'-'*35}  {'-'*55}")
        for proj_title, cids in test_projects:
            # Leave last paper out, query with rest
            query_cids = cids[:-1]
            expect_cid = cids[-1]

            query = " ".join("P" + c for c in query_cids[:15])
            preds = paper_model.predict(query, k=k)
            if not preds:
                continue
            pred_cids = {lbl[1:] for lbl, _ in preds}
            ok = expect_cid in pred_cids
            if ok:
                hits += 1
            mark = "OK" if ok else "MISS"
            p_str = proj_title[:35]
            t_str = paper_titles.get(expect_cid, "?")[:55]
            print(f"  {mark:5s} {p_str:35s}  {t_str}")

        pct = hits / len(test_projects) if test_projects else 0
        print(f"\n  {hits}/{len(test_projects)} held-out papers found"
              f" ({pct:.0%})")

    def test_recommend_for_project(self, paper_model, all_projects,
                                   paper_titles):
        """Per-source voting: query each source paper, aggregate votes."""
        rng = random.Random(42)
        test_projects = [
            (t, c) for t, c in all_projects if len(c) >= 5]
        if len(test_projects) > 8:
            test_projects = rng.sample(test_projects, 8)

        got_recs = 0
        k_per_source = 30
        k_show = 5
        print(f"\n  Project recommendations (per-source voting,"
              f" top {k_show}, excluding sources):")
        for proj_title, source_cids in test_projects:
            source_titles = [
                paper_titles.get(c, "?")[:45] for c in source_cids[:3]]
            print(f"\n  Project: {proj_title}")
            print(f"  Sources ({len(source_cids)}): "
                  + " | ".join(source_titles))

            source_set = set(source_cids)

            # Query each source paper independently, aggregate votes
            votes: Counter[str] = Counter()
            scores: dict[str, float] = {}
            for cid in source_cids:
                preds = paper_model.predict("P" + cid, k=k_per_source)
                for lbl, score in preds:
                    if lbl[1:] not in source_set:
                        votes[lbl] += 1
                        scores[lbl] = max(scores.get(lbl, 0.0), score)

            # Rank by vote count, break ties by best score
            ranked = sorted(votes.keys(),
                            key=lambda l: (votes[l], scores[l]),
                            reverse=True)
            if ranked:
                got_recs += 1
                print(f"  Recommended:")
                for lbl in ranked[:k_show]:
                    cid = lbl[1:]
                    t = paper_titles.get(cid, "?")[:60]
                    print(f"    {votes[lbl]:2d} votes  {t}"
                          f"  ({scores[lbl]:.2f})")
            else:
                print(f"  (no novel recommendations)")

        print(f"\n  {got_recs}/{len(test_projects)} projects got"
              f" recommendations")


class TestSpeed:
    def test_training_speed(self):
        """Measure training speed on paper data."""
        path = _data_cache.get("path")
        if not path or not os.path.isfile(path):
            _data_cache.clear()
            path, _, _, _, _ = _get_paper_data()
        t0 = time.time()
        StarSpace.train(
            path, dim=100, epoch=5, lr=0.01,
            train_mode=1, bucket=100000, verbose=0,
            neg_search_limit=50, max_neg_samples=10,
            label="P", min_count=1)
        elapsed = time.time() - t0
        print(f"\n  Papers 5 epochs: {elapsed:.1f}s")
        assert elapsed < 180, f"Training took {elapsed:.1f}s (expected <180s)"
