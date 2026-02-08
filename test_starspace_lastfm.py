"""Last.FM dataset tests — artist recommendation and tag co-occurrence.

Two test suites using the hetrec2011-lastfm-2k dataset:

1. Artist recommendation (mode 1): label-only data with custom prefix "A",
   replicating examples/recomm_user_artists.sh. Collaborative filtering.

2. Tag co-occurrence (mode 1): each artist's user-assigned tags as labels
   with prefix "T". Holistic tag context predicts related tags.

Dataset: hetrec2011-lastfm-2k (1892 users, 17632 artists, 11946 tags)
"""

import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from starspace import StarSpace

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "lastfm"
ARTIST_TRAIN_FILE = str(DATA_DIR / "user_artists.train")
ARTIST_NAMES_FILE = str(DATA_DIR / "artists.dat")
TAG_NAMES_FILE = str(DATA_DIR / "tags.dat")
USER_TAGS_FILE = str(DATA_DIR / "user_taggedartists.dat")

# Skip entire module if dataset not downloaded
pytestmark = pytest.mark.skipif(
    not os.path.isfile(ARTIST_TRAIN_FILE),
    reason="Last.FM dataset not found (download to data/lastfm/)")


def _load_artist_names():
    """Load artist ID → name mapping from artists.dat."""
    names = {}
    if not os.path.isfile(ARTIST_NAMES_FILE):
        return names
    with open(ARTIST_NAMES_FILE, encoding="utf-8", errors="replace") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                names["A" + parts[0]] = parts[1]
    return names


ARTIST_NAMES = _load_artist_names()


# ── Artist recommendation (mode 1) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def artist_model():
    """Train on Last.FM artist data, mode 1 (examples/recomm_user_artists.sh)."""
    return StarSpace.train(
        ARTIST_TRAIN_FILE, dim=100, epoch=100, lr=0.01,
        train_mode=1, bucket=0, verbose=0,
        neg_search_limit=100, max_neg_samples=100,
        label="A", init_rand_sd=0.01)


class TestArtistRecommendation:
    def test_learns_artists(self, artist_model):
        """Should learn >1000 artist labels."""
        print(f"\n  Artists learned: {artist_model.vocab.nlabels}")
        assert artist_model.vocab.nlabels > 1000

    def test_no_words(self, artist_model):
        """Label-only data should have no word tokens."""
        assert artist_model.vocab.nwords == 0

    def test_predict_returns_labels(self, artist_model):
        """predict() with artist labels returns artist recommendations."""
        preds = artist_model.predict("A51 A52 A53", k=10)
        assert len(preds) > 0
        for lbl, _ in preds:
            assert lbl.startswith("A")

    def test_semantic_recommendations(self, artist_model):
        """Recommends genre-appropriate artists."""
        queries = {
            # Pop: Britney + Rihanna + Katy Perry
            #   → Christina / Beyoncé / Madonna / P!nk
            "pop": ("A289 A288 A300",
                    {"A292", "A295", "A67", "A302", "A461", "A55"}),
            # Classic rock: Beatles + Pink Floyd + Queen
            #   → Led Zep / AC/DC / Bowie
            "classic_rock": ("A227 A163 A959",
                             {"A1412", "A706", "A599", "A511"}),
            # Alt/indie: Muse + Radiohead + Coldplay
            #   → Killers / Arctic Monkeys / Oasis
            "alt_indie": ("A190 A154 A65",
                          {"A229", "A207", "A228", "A533", "A173"}),
            # Metal/punk: Linkin Park + Metallica + Green Day
            #   → Nirvana / 30STM / Paramore
            "metal_punk": ("A377 A707 A230",
                           {"A234", "A486", "A498", "A378", "A220"}),
        }
        hits = 0
        details = []
        k = 20
        print(f"\n  Artist recommendations (top {k}):")
        print(f"  {'':4s} {'genre':<12s}  {'query':40s}  {'result'}")
        print(f"  {'':4s} {'-'*12}  {'-'*40}  {'-'*50}")
        for genre, (query, expect) in queries.items():
            preds = artist_model.predict(query, k=k)
            top_ids = [lbl for lbl, _ in preds]
            found = set(top_ids) & expect
            ok = len(found) > 0
            if ok:
                hits += 1
            query_str = ", ".join(
                ARTIST_NAMES.get(a, a) for a in query.split())
            top_str = ", ".join(
                ARTIST_NAMES.get(a, a) for a in top_ids[:5])
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} {genre:<12s}  {query_str:40s}  {top_str}")
            details.append(
                f"  {mark:4s} {genre}: "
                f"query=[{query_str}] top5=[{top_str}]")
        print(f"  {hits}/{len(queries)} genre hits")
        assert hits >= 3, (
            f"{hits}/{len(queries)} genre hits "
            f"(need 3):\n" + "\n".join(details))


# ── Tag co-occurrence (mode 1) ──────────────────────────────────────────────

def _build_tag_data():
    """Build tag co-occurrence train file from user_taggedartists.dat.

    Returns path to a temp file. Caller must clean up.
    """
    # Load tag names
    tag_names = {}
    with open(TAG_NAMES_FILE, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                tag_names[parts[0]] = parts[1].strip().replace(" ", "_").lower()

    # Build per-artist tag sets
    artist_tags: dict[str, set[str]] = defaultdict(set)
    with open(USER_TAGS_FILE, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                tag = tag_names.get(parts[2], "")
                if tag:
                    artist_tags[parts[1]].add(tag)

    # Keep tags appearing on 5+ artists
    tag_freq: Counter[str] = Counter()
    for tags in artist_tags.values():
        for t in tags:
            tag_freq[t] += 1
    frequent_tags = {t for t, c in tag_freq.items() if c >= 5}

    # Each line = artist's tags (label prefix T), min 3 tags
    rng = random.Random(42)
    lines = []
    for tags in artist_tags.values():
        filtered = [t for t in tags if t in frequent_tags]
        if len(filtered) >= 3:
            lines.append(" ".join("T" + t for t in sorted(filtered)))

    rng.shuffle(lines)

    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ss_tags_")
    for line in lines:
        f.write(line + "\n")
    f.close()
    return f.name


@pytest.fixture(scope="module")
def tag_model():
    """Train Python model on tag co-occurrence data, mode 1."""
    train_path = _build_tag_data()
    model = StarSpace.train(
        train_path, dim=50, epoch=20, lr=0.01,
        train_mode=1, bucket=0, verbose=0,
        neg_search_limit=50, max_neg_samples=10, label="T")
    os.unlink(train_path)
    return model


class TestTagCooccurrence:
    def test_learns_tags(self, tag_model):
        """Should learn >1000 tag labels."""
        print(f"\n  Tags learned: {tag_model.vocab.nlabels}")
        assert tag_model.vocab.nlabels > 1000

    def test_no_words(self, tag_model):
        """Tag-only data should have no word tokens."""
        assert tag_model.vocab.nwords == 0

    def test_semantic_tags(self, tag_model):
        """Tag queries return genre-appropriate co-occurring tags."""
        queries = {
            # metal + thrash_metal → heavy_metal, speed_metal, death_metal
            "metal": ("Tmetal Tthrash_metal",
                      {"Theavy_metal", "Tspeed_metal", "Tdeath_metal",
                       "Thard_rock"}),
            # pop + dance → electronic, female_vocalists, electro
            "pop": ("Tpop Tdance",
                    {"Telectronic", "Tfemale_vocalists", "Telectro",
                     "Thouse"}),
            # electronic + new_wave + 80s → synthpop, post-punk
            "synth": ("Telectronic Tnew_wave T80s",
                      {"Tsynthpop", "Tsynth_pop", "Tpost-punk",
                       "Tbritish"}),
            # hip-hop + rap → hip_hop, rnb, underground
            "hiphop": ("Thip-hop Trap",
                       {"Thip_hop", "Trnb", "Tunderground_hip-hop",
                        "Thip_hop/rap"}),
            # classic_rock + 60s → british, psychedelic, oldies
            "classic": ("Tclassic_rock Trock T60s",
                        {"Tbritish", "Toldies", "T70s", "Tpsychedelic",
                         "Thard_rock"}),
        }
        hits = 0
        details = []
        k = 10
        print(f"\n  Tag co-occurrence (top {k}):")
        print(f"  {'':4s} {'genre':<10s}  {'query':30s}  {'result'}")
        print(f"  {'':4s} {'-'*10}  {'-'*30}  {'-'*50}")
        for genre, (query, expect) in queries.items():
            preds = tag_model.predict(query, k=k)
            top_ids = [lbl for lbl, _ in preds]
            found = set(top_ids) & expect
            ok = len(found) > 0
            if ok:
                hits += 1
            query_str = ", ".join(t[1:] for t in query.split())
            top_str = ", ".join(t[1:] for t in top_ids[:5])
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} {genre:<10s}  {query_str:30s}  {top_str}")
            details.append(
                f"  {mark:4s} {genre}: "
                f"query=[{query_str}] top5=[{top_str}]")
        print(f"  {hits}/{len(queries)} genre hits")
        assert hits >= 4, (
            f"{hits}/{len(queries)} genre hits "
            f"(need 4):\n" + "\n".join(details))


# ── Speed ───────────────────────────────────────────────────────────────────

class TestSpeed:
    def test_training_speed(self):
        """Measure Python training speed on Last.FM artist data."""
        t0 = time.time()
        StarSpace.train(
            ARTIST_TRAIN_FILE, dim=100, epoch=20, lr=0.01,
            train_mode=1, bucket=0, verbose=0,
            neg_search_limit=100, max_neg_samples=100,
            label="A", init_rand_sd=0.01)
        elapsed = time.time() - t0
        print(f"\n  Last.FM 20 epochs: {elapsed:.1f}s")
        assert elapsed < 60, f"Training took {elapsed:.1f}s (expected <60s)"
