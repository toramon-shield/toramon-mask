#!/usr/bin/env python3
"""HF 形式 OpenAIPrivacyFilter チェックポイントの追加学習（コンテナ内で実行）。

  python -m toramon.train.finetune_hf --base <hf_dir> --train a.jsonl [b.jsonl] --dev dev.jsonl --out runs/<name> \
      --mode experts_frozen|lora|full --epochs 3 --lr 1e-4 --batch-size 8 --max-length 512 --seed 0

- 入力は Toramon JSONL（entities に action）。action=MASK かつ基本 8 カテゴリの span だけを教師にする。KEEP/REVIEW は O。
- ラベル空間は base のものを維持する（S 系は alias: private_email→EMAIL 等、label_mapping.yaml の逆写像）。
- mode:
    experts_frozen: MoE experts と埋め込みを凍結（sumeshi 方式、約 90M 更新）
    lora:           attention q/k/v/o と router に LoRA（peft）、分類ヘッドは全更新
    full:           全パラメータ更新（メモリを測ってから）
- 長い入力は span を保ちながらトークン列を分割（max_length ごと、span 途中で切れる場合は直前の境界で切る）。
- 保存: out/model/（HF 形式）、out/train_summary.json（設定・seed・時間・ピークメモリ・学習可能パラメータ数・loss 履歴）、out/dev_eval.json。
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
COMMON = ["private_person", "private_address", "private_email", "private_phone", "private_date", "account_number", "private_url", "secret"]
# S 系ラベル空間（OpenMed 225）で各共通カテゴリをどの既存行で学習するか（sumeshi finetune_jp.py の alias と同じ）
S_ALIAS = {"account_number": ["account_number", "BANKACCOUNT", "ACCOUNTNAME"], "private_date": ["private_date", "DATE", "DATEOFBIRTH"],
           "private_email": ["private_email", "EMAIL"], "private_phone": ["private_phone", "PHONE"], "private_url": ["private_url", "URL"],
           "secret": ["secret", "PASSWORD"], "private_person": ["private_person"], "private_address": ["private_address"]}


def resolve_category(cat: str, label2id: dict) -> str | None:
    for c in S_ALIAS.get(cat, [cat]):
        if all(f"{t}-{c}" in label2id for t in "BIES"):
            return c
    return None


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_row(row: dict, tok, label2id: dict, max_length: int) -> list[dict]:
    """1 文書 → 1 つ以上の学習例（span を跨がない位置で分割）。"""
    text = row["text"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    labels = [label2id["O"]] * n
    spans = [e for e in row.get("entities", []) if e.get("action") == "MASK" and e["label"] in COMMON]
    span_tok_ranges = []
    for e in spans:
        cat = resolve_category(e["label"], label2id)
        if cat is None:
            continue
        idx = [i for i, (s, t) in enumerate(offs) if s != t and s < e["end"] and e["start"] < t]
        if not idx:
            continue
        if len(idx) == 1:
            labels[idx[0]] = label2id[f"S-{cat}"]
        else:
            labels[idx[0]] = label2id[f"B-{cat}"]; labels[idx[-1]] = label2id[f"E-{cat}"]
            for i in idx[1:-1]:
                labels[i] = label2id[f"I-{cat}"]
        span_tok_ranges.append((idx[0], idx[-1]))
    # 分割
    out = []
    start = 0
    while start < n:
        end = min(n, start + max_length)
        if end < n:
            # span 途中なら手前の境界へ
            for a, b in span_tok_ranges:
                if a < end <= b:
                    end = a
                    break
            if end <= start:  # 1 span が max_length を超える: 強制分割（後半の正解は次チャンクで保持）
                end = min(n, start + max_length)
        out.append({"input_ids": ids[start:end], "labels": labels[start:end]})
        start = end
    return out


class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch, pad_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"]); ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def set_trainable(model, mode: str, lora_r: int = 16, lora_alpha: int = 32):
    if mode == "full":
        for p in model.parameters(): p.requires_grad_(True)
        return model
    if mode == "experts_frozen":
        for n, p in model.named_parameters():
            p.requires_grad_(not (".mlp.experts." in n or n == "model.embed_tokens.weight"))
        return model
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        # router は独自モジュール（peft 非対応）なので LoRA 対象から外し、router と分類ヘッド・norm は全更新にする
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"])
        model = get_peft_model(model, cfg)
        for n, p in model.named_parameters():
            if ".mlp.router." in n or "layernorm" in n or n.endswith("model.norm.weight") or ".score." in n:
                p.requires_grad_(True)
        return model
    raise ValueError(mode)


def _eval_epoch(model, tok, a, out, ep: int, label_space: str) -> dict:
    """epoch 末の dev 評価: スナップショットを保存し、アダプタで再読込して評価（学習時と同じ復号条件）。lora はマージ済みスナップショットを保存する。"""
    sys.path.insert(0, str(ROOT / "src"))
    from toramon.adapters.hf_token_cls import HFTokenClassifier
    from toramon.evaluate import load_gold_jsonl, score_doc, aggregate
    snap = out / f"epoch{ep}" / "model"; snap.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    if a.mode == "lora":
        # LoRA: モデルの複製に merge_and_unload を適用して HF 形式で保存する（学習中のモデルは触らない）。
        # merge_adapter + get_base_model().save_pretrained では modules_to_save（分類ヘッド score）が
        # score.modules_to_save.default.* の名前で保存され、再読込時にヘッドが再初期化されて F1 が 0 になる。
        import copy
        merged = copy.deepcopy(model).merge_and_unload()
        merged.save_pretrained(snap, safe_serialization=True)
        del merged; torch.cuda.empty_cache()
    else:
        model.save_pretrained(snap, safe_serialization=True)
    tok.save_pretrained(snap)
    clf = HFTokenClassifier(str(snap), device="cuda", label_space=label_space, threshold=a.eval_threshold)
    docs = load_gold_jsonl(a.dev)
    for d in docs: d.pred = clf.detect(d.text)
    res = aggregate([score_doc(d) for d in docs])
    (snap.parent / "dev_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    del clf; torch.cuda.empty_cache()
    return {"epoch": ep, "dev_micro_f1": res["micro"]["f1"], "dev_macro_f1": res["macro_f1"], "full_coverage": res["full_coverage_rate"],
            "pii_remaining": res["pii_remaining_doc_rate"], "nontarget": res["nontarget_char_mask_rate"], "neg_fp": res["negative_doc_fp_rate"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--dev")
    ap.add_argument("--out", required=True); ap.add_argument("--mode", default="experts_frozen", choices=["experts_frozen", "lora", "full"])
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1); ap.add_argument("--max-length", type=int, default=512); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.05); ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-space", default="auto", choices=["auto", "identity", "S"]); ap.add_argument("--eval-threshold", type=float, default=0.5)
    ap.add_argument("--max-train-rows", type=int, default=0); ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--pos-weight", type=float, default=1.0, help="O 以外のクラスの損失重み（>1 で再現率寄り）")
    ap.add_argument("--eval-each-epoch", action="store_true", help="各 epoch 終了時に dev を評価し、最良 epoch を out/model に残す（飽和判定用）")
    ap.add_argument("--patience", type=int, default=2, help="--eval-each-epoch 時、dev F1 が改善しない epoch 数がこの値に達したら打ち切る")
    ap.add_argument("--min-delta", type=float, default=0.001, help="改善とみなす dev F1 の最小増分")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    out = Path(a.out); (out / "model").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForTokenClassification.from_pretrained(a.base, dtype=torch.bfloat16)
    label2id = {v: int(k) for k, v in model.config.id2label.items()} if isinstance(next(iter(model.config.id2label)), str) else {v: k for k, v in model.config.id2label.items()}
    n_labels = len(label2id)
    label_space = a.label_space if a.label_space != "auto" else ("S" if n_labels > 33 else "identity")
    model = set_trainable(model, a.mode, a.lora_r)
    model.to("cuda")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad); n_all = sum(p.numel() for p in model.parameters())

    rows = load_rows(a.train)
    if a.max_train_rows: rows = rows[: a.max_train_rows]
    items = [ex for r in rows for ex in encode_row(r, tok, label2id, a.max_length)]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else model.config.pad_token_id
    dl = DataLoader(DS(items), batch_size=a.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id), generator=torch.Generator().manual_seed(a.seed))
    steps_per_epoch = math.ceil(len(dl) / a.grad_accum); total_steps = max(1, int(steps_per_epoch * a.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    warm = int(total_steps * a.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total_steps - s) / max(1, total_steps - warm)) if s >= warm else (s + 1) / max(1, warm))

    summary = {"base": a.base, "mode": a.mode, "pos_weight": a.pos_weight, "label_space": label_space, "n_labels": n_labels, "trainable_params": n_train, "total_params": n_all,
               "train_files": a.train, "n_rows": len(rows), "n_examples": len(items), "epochs": a.epochs, "lr": a.lr, "batch_size": a.batch_size,
               "grad_accum": a.grad_accum, "max_length": a.max_length, "seed": a.seed, "total_steps": total_steps,
               "started_at": datetime.datetime.now().astimezone().isoformat(), "loss_history": []}
    print(json.dumps({k: summary[k] for k in ("mode", "label_space", "n_labels", "trainable_params", "total_params", "n_rows", "n_examples", "total_steps")}), flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); step = 0; model.train(); done = False
    ep = 0
    while not done:
        for i, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mo = model(**{k: v for k, v in batch.items() if k != "labels"})
                logits = mo.logits.float()
                if a.pos_weight != 1.0:
                    w = torch.full((logits.shape[-1],), a.pos_weight, device=logits.device); w[label2id["O"]] = 1.0
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1), weight=w, ignore_index=-100)
                else:
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1), ignore_index=-100)
            (loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or step == 1:
                    summary["loss_history"].append({"step": step, "loss": float(loss), "lr": sched.get_last_lr()[0], "t": round(time.perf_counter() - t0, 1)})
                    print(f"step {step}/{total_steps} loss {float(loss):.4f} lr {sched.get_last_lr()[0]:.2e} {time.perf_counter()-t0:.0f}s", flush=True)
                if step >= total_steps: done = True; break
        ep += 1
        if a.eval_each_epoch and a.dev:
            m = _eval_epoch(model, tok, a, out, ep, label_space)
            summary.setdefault("epoch_evals", []).append(m)
            best = summary.get("best_epoch")
            if best is None or m["dev_micro_f1"] > best["dev_micro_f1"] + a.min_delta:
                summary["best_epoch"] = m; bad = 0
            else:
                bad = summary.get("bad_epochs", 0) + 1
            summary["bad_epochs"] = bad
            print(json.dumps({"epoch": ep, **{k: m[k] for k in ("dev_micro_f1", "full_coverage", "pii_remaining")}, "best_epoch": summary["best_epoch"]["epoch"], "bad_epochs": bad}), flush=True)
            if bad >= a.patience:
                print(f"early stop at epoch {ep} (patience {a.patience})", flush=True); done = True
            model.train()
        if ep > math.ceil(a.epochs) + 1: break
    summary["train_seconds"] = round(time.perf_counter() - t0, 1)
    summary["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # 保存（LoRA はマージして HF 形式で保存）
    if a.mode == "lora":
        model = model.merge_and_unload()
    model.eval()
    if a.eval_each_epoch and a.dev and summary.get("best_epoch"):
        import shutil
        be = summary["best_epoch"]["epoch"]; src = out / f"epoch{be}" / "model"
        if (out / "model").exists(): shutil.rmtree(out / "model")
        shutil.copytree(src, out / "model"); summary["saved_to"] = str(out / "model"); summary["saved_from_epoch"] = be
        for d in out.glob("epoch*"):
            if d.name != f"epoch{be}": shutil.rmtree(d, ignore_errors=True)  # 最良以外の epoch スナップショットは削除（容量）
    else:
        model.save_pretrained(out / "model", safe_serialization=True); tok.save_pretrained(out / "model")
        summary["saved_to"] = str(out / "model")

    # 再読込して dev 評価
    if a.dev:
        sys.path.insert(0, str(ROOT / "src"))
        from toramon.adapters.hf_token_cls import HFTokenClassifier
        from toramon.evaluate import load_gold_jsonl, score_doc, aggregate
        del model; torch.cuda.empty_cache()
        clf = HFTokenClassifier(str(out / "model"), device="cuda", label_space=label_space, threshold=a.eval_threshold)
        docs = load_gold_jsonl(a.dev)
        preds = []
        lat = []
        with open(out / "dev_pred.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                t1 = time.perf_counter(); d.pred = clf.detect(d.text); lat.append((time.perf_counter() - t1) * 1000)
                f.write(json.dumps({"id": d.id, "spans": [s.to_dict() for s in d.pred]}, ensure_ascii=False) + "\n")
        res = aggregate([score_doc(d) for d in docs])
        res["latency_ms_p50"] = sorted(lat)[len(lat) // 2]
        (out / "dev_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["dev_micro_f1"] = res["micro"]["f1"]; summary["dev_full_coverage"] = res["full_coverage_rate"]
        print(json.dumps({"dev_micro_f1": res["micro"]["f1"], "dev_macro_f1": res["macro_f1"], "full_coverage": res["full_coverage_rate"], "pii_remaining": res["pii_remaining_doc_rate"]}), flush=True)
    summary["finished_at"] = datetime.datetime.now().astimezone().isoformat()
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
