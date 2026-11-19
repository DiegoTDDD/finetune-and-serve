"""Download FiNER-139 and build the task's train, validation and test sets.

- concepts: the N most frequent XBRL tags in the official training split;
- a sentence is in scope if all its facts use those concepts; sentences with no facts are
  kept as negatives (a share of each set), because most numbers in filings are not tagged
  and a model must learn to leave them alone;
- each set is sampled from the matching official split, so test sentences come from filings
  the model never trained on.

    python scripts/prepare_data.py --out data
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftserve.finer import Example, in_scope, parse_labels, tag_counts, to_example  # noqa: E402

REPO = "nlpaueb/finer-139"
MAX_CHARS = 400


def read_split(zf: zipfile.ZipFile, name: str, labels: list[str]) -> list[Example]:
    member = next(n for n in zf.namelist() if n.endswith(f"{name}.jsonl"))
    with zf.open(member) as raw:
        return [to_example(name, json.loads(line), labels) for line in io.TextIOWrapper(raw, "utf-8")]


def sample(pool: list[Example], tags: set[str], n: int, neg_frac: float, rng: random.Random) -> list[Example]:
    short = [e for e in pool if len(e.text) <= MAX_CHARS and len(e.text) >= 40]
    pos = sorted((e for e in short if e.facts and in_scope(e, tags)), key=lambda e: e.id)
    neg = sorted((e for e in short if not e.facts), key=lambda e: e.id)
    n_neg = round(n * neg_frac)
    out = rng.sample(pos, n - n_neg) + rng.sample(neg, n_neg)
    rng.shuffle(out)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--tags", type=int, default=12)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--neg-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(REPO).sha
    script = Path(hf_hub_download(REPO, "finer-139.py", repo_type="dataset", revision=revision))
    archive = Path(hf_hub_download(REPO, "finer139.zip", repo_type="dataset", revision=revision))
    labels = parse_labels(script.read_text(encoding="utf-8"))

    with zipfile.ZipFile(archive) as zf:
        splits = {s: read_split(zf, s, labels) for s in ("train", "validation", "test")}
    counts = tag_counts(splits["train"])
    tags = [t for t, _ in counts.most_common(args.tags)]
    tagset = set(tags)

    rng = random.Random(args.seed)
    sets = {
        "train": sample(splits["train"], tagset, args.train, args.neg_frac, rng),
        "val": sample(splits["validation"], tagset, args.val, args.neg_frac, rng),
        "test": sample(splits["test"], tagset, args.test, args.neg_frac, rng),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tags.json").write_text(json.dumps(tags, indent=1) + "\n", encoding="utf-8")
    files = {"tags.json": sha256(args.out / "tags.json")}
    for name, rows in sets.items():
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(e.to_json()) + "\n" for e in rows), encoding="utf-8")
        files[path.name] = sha256(path)
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": revision,
        "license": "CC BY-SA 4.0",
        "raw_sentences": {k: len(v) for k, v in splits.items()},
        "tags": {t: counts[t] for t in tags},
        "params": vars(args) | {"out": str(args.out), "max_chars": MAX_CHARS},
        "sets": {
            k: {
                "sentences": len(v),
                "facts": sum(len(e.facts) for e in v),
                "no_facts": sum(not e.facts for e in v),
            }
            for k, v in sets.items()
        },
        "sha256": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
