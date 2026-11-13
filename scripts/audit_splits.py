#!/usr/bin/env python3
"""分割間のデータ漏洩・偏り監査。

検査: 完全一致 / NFKC 正規化一致 / 骨格一致（固有値を {{KIND}} に戻した text）/ 文字 5-gram Jaccard ≥ 0.8 の近似重複 /
     scenario_family・template_family・value_pool の split 越境 / 生成元の偏り / 未レビュー例の混入 / スキーマ・位置検査。
  python scripts/audit_splits.py data/train/*.jsonl data/dev/*.jsonl [--test data/locked/test.jsonl] --out runs/audit/audit.json
"""
from __future__ import annotations
import argparse, json, sys, unicodedata, hashlib
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.spans import Span, validate_spans  # noqa: E402

LABEL_KIND = {"private_person": "PERSON", "private_address": "ADDRESS", "private_email": "EMAIL", "private_phone": "PHONE", "private_date": "DATE",
              "account_number": "ACCOUNT", "private_url": "URL", "secret": "SECRET"}


def skeleton(text: str, ents: list[dict]) -> str:
    out = []; cur = 0
    for e in sorted(ents, key=lambda e: e["start"]):
        out.append(text[cur:e["start"]]); out.append("{{" + LABEL_KIND.get(e["label"], e["label"]) + "}}"); cur = e["end"]
    out.append(text[cur:]); return "".join(out)


def ngrams(s: str, n: int = 5) -> set:
    s = unicodedata.normalize("NFKC", s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out"); ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--max-pairs", type=int, default=2_000_000)
    a = ap.parse_args()
    docs = []
    for p in a.files:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                docs.append(json.loads(line))
    rep: dict = {"n_docs": len(docs), "by_split": Counter(d["split"] for d in docs), "problems": []}
    # 位置・スキーマ簡易検査
    bad = 0
    for d in docs:
        pr = validate_spans(d["text"], [Span(e["start"], e["end"], e["label"], e.get("text", ""), e.get("action", "MASK")) for e in d.get("entities", [])])
        if pr:
            bad += 1; rep["problems"].append({"id": d["id"], "span_problems": pr[:3]})
    rep["docs_with_span_problems"] = bad
    # 越境
    for key in ("scenario_family", "template_family", "value_pool"):
        m = defaultdict(set)
        for d in docs:
            if d.get(key): m[d[key]].add(d["split"])
        cross = {k: sorted(v) for k, v in m.items() if len(v) > 1 and not (v <= {"pilot", "train", "dev"} and key == "value_pool")}
        # pilot は開発側専用データ（train/dev family を含む）。pilot と dev/train の重なりは警告扱い（pilot を学習に使う場合は dev family の文書を除外する）。test との重なりは常にエラー。
        warn = {k: v for k, v in cross.items() if "pilot" in v and "test" not in v}
        cross = {k: v for k, v in cross.items() if k not in warn}
        rep[f"{key}_cross_split"] = cross
        if warn:
            rep[f"{key}_pilot_overlap_warning"] = warn
    # 重複
    exact = defaultdict(list); norm = defaultdict(list); skel = defaultdict(list)
    for d in docs:
        exact[hashlib.sha1(d["text"].encode()).hexdigest()].append(d)
        norm[hashlib.sha1(unicodedata.normalize("NFKC", d["text"]).encode()).hexdigest()].append(d)
        skel[hashlib.sha1(skeleton(d["text"], d.get("entities", [])).encode()).hexdigest()].append(d)
    def cross_split_groups(groups):
        return [[x["id"] for x in g] for g in groups.values() if len({x["split"] for x in g}) > 1]
    rep["exact_dup_cross_split"] = cross_split_groups(exact)
    rep["normalized_dup_cross_split"] = cross_split_groups(norm)
    rep["skeleton_dup_cross_split"] = cross_split_groups(skel)
    rep["skeleton_dup_within_split"] = sum(1 for g in skel.values() if len(g) > 1)
    # 近似重複（split をまたぐ組のみ、n-gram Jaccard）
    by_split = defaultdict(list)
    for d in docs: by_split[d["split"]].append((d["id"], ngrams(d["text"])))
    near = []; pairs = 0
    splits = sorted(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            for ida, ga in by_split[splits[i]]:
                for idb, gb in by_split[splits[j]]:
                    pairs += 1
                    if pairs > a.max_pairs: break
                    if not ga or not gb: continue
                    jac = len(ga & gb) / len(ga | gb)
                    if jac >= a.jaccard: near.append({"a": ida, "b": idb, "jaccard": round(jac, 3)})
    rep["near_dup_cross_split"] = near; rep["near_dup_pairs_checked"] = pairs
    # 生成元偏り・レビュー状態
    rep["generator_by_split"] = {s: Counter(d.get("generator_model") for d in docs if d["split"] == s) for s in by_split}
    rep["review_status_by_split"] = {s: Counter(d.get("review_status") for d in docs if d["split"] == s) for s in by_split}
    rep["negative_type_by_split"] = {s: Counter(d.get("negative_type") for d in docs if d["split"] == s) for s in by_split}
    rep["ok"] = not (bad or rep["exact_dup_cross_split"] or rep["normalized_dup_cross_split"] or rep["skeleton_dup_cross_split"] or near or any(rep[f"{k}_cross_split"] for k in ("scenario_family", "template_family", "value_pool")))
    js = json.dumps(rep, ensure_ascii=False, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(js, encoding="utf-8")
    print(js[:3000])
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
