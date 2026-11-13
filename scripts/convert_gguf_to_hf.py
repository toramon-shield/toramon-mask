#!/usr/bin/env python3
"""Reconstruct a Hugging Face ``OpenAIPrivacyFilterForTokenClassification`` checkpoint
from the sumeshi privacy-filter-jp GGUF (inverse of
third_party/privacy-filter-jp.cpp/scripts/convert.py).

Purpose: make the Japanese privacy-filter model trainable / fine-tunable with
transformers. The author published only the GGUF (F16 weight matrices, F32
norms/biases/router/sinks), so this is a RECONSTRUCTION, not the author's
original bf16 checkpoint:

  * F16 -> BF16 is NOT lossless in general. F16 keeps 10 mantissa bits, BF16
    keeps 7, so a weight-matrix element (embeddings, q/k/v/o, expert gate/up/down,
    score) that uses those extra bits is re-rounded (round-to-nearest-even,
    relative error up to 2^-8). If the author's original checkpoint was bf16,
    bf16 -> f16 -> bf16 is exact only for |x| in f16's normal range
    (6.1e-5 .. 65504); smaller values were quantized onto the f16 subnormal grid
    (or flushed to 0 below 6e-8) by the author's export and cannot be recovered.
    The script measures this per tensor and records it in manifest.json
    (``lossless`` = every element bf16-exact, ``n_f16_subnormal`` = elements
    with 0 < |x| < 6.1e-5). For the published privacy-filter-jp-f16.gguf every
    F16 element turned out to be bf16-exact (0 re-rounded), i.e. the bf16 file is
    a lossless re-encoding of the GGUF; equality with the author's original
    checkpoint still cannot be proven from the GGUF alone (subnormal caveat).
    ``--weights-dtype f16`` keeps those tensors as float16 instead (always
    lossless w.r.t. the GGUF, but a non-standard dtype for this architecture).
  * The GGUF's F32 tensors (norms, biases, router weight/bias) are checked to be
    exactly bf16-representable (low 16 mantissa bits zero) before being stored
    as bf16; the conversion of those is lossless. ``self_attn.sinks`` is kept
    float32, as in the reference checkpoint.
  * Parameter names / shapes / dtypes mirror the OpenAI reference checkpoint
    (openai/privacy-filter): model.embed_tokens.weight, model.layers.N.self_attn.
    {q,k,v,o}_proj.{weight,bias}, self_attn.sinks, mlp.router.{weight,bias},
    mlp.experts.{gate_up_proj,gate_up_proj_bias,down_proj,down_proj_bias},
    {input_layernorm,post_attention_layernorm}.weight, model.norm.weight,
    score.{weight,bias}.

Tensor transforms (exact inverses of convert.py; the parity test
scripts/verify_gguf_hf_parity.py is the guard, since the expert matrices are
square and a wrong transpose is shape-valid):
  * ffn_down_exps.weight [E, out, in]            -> down_proj = transpose(-1,-2) -> [E, in, out]
  * ffn_gate_exps / ffn_up_exps [E, inter, in]   -> concat on dim 1 (gate first, then up;
                                                   CONCATENATED halves, not interleaved)
                                                   -> transpose(-1,-2) -> gate_up_proj [E, in, 2*inter]
  * ffn_gate_exps.bias / ffn_up_exps.bias [E, inter] -> concat on last dim -> gate_up_proj_bias [E, 2*inter]
  * everything else is a straight copy (GGUFReader already yields row-major numpy
    arrays in HF orientation, e.g. token_embd [vocab, d_model]).

Tokenizer: the GGUF carries the o200k BPE vocabulary (gpt2 model / gpt-4o pre),
identical to the OpenAI reference checkpoint's tokenizer.json, which is copied
along with tokenizer_config.json (the parity script verifies token ids).

Runs on the host venv (numpy + gguf + safetensors; no torch needed - the
safetensors file is written directly, the format is trivial) or in the container.

Usage:
  .venv/bin/python scripts/convert_gguf_to_hf.py \
      --gguf models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf \
      --ref  models/openai-privacy-filter \
      --out  models/sumeshi-privacy-filter-jp-hf-from-f16
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ARCH = "openai-privacy-filter"
EXPECTED_GGUF_SHA256 = "e2bafc05ef7e6beb354e78cd77c8cfb5101d55044f23b2cdf343731a882f3b1c"

ST_DTYPE = {"bf16": "BF16", "f16": "F16", "f32": "F32"}
CFG_DTYPE = {"bf16": "bfloat16", "f16": "float16"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 (uint16 payload), round-to-nearest-even. No NaN/inf expected."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding_bias) >> 16).astype(np.uint16)


def is_bf16_exact(x: np.ndarray) -> bool:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return not bool(np.any(u & 0xFFFF))


def read_ref_header(ref_dir: Path) -> dict:
    with open(ref_dir / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def gguf_labels(reader) -> list[str]:
    f = reader.fields[f"{ARCH}.classifier.output_labels"]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def gguf_scalar(reader, key: str):
    f = reader.fields[key]
    v = f.parts[f.data[0]]
    if f.types and f.types[0].name == "STRING":
        return bytes(v).decode("utf-8")
    return v[0].item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="OpenAI reference HF checkpoint dir (config/tokenizer template)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights-dtype", choices=("bf16", "f16"), default="bf16",
                    help="dtype for the GGUF F16 weight matrices (default bf16 = mirror the reference; f16 = lossless)")
    ap.add_argument("--skip-gguf-sha", action="store_true", help="do not hash the 2.8 GB source (saves ~10 s)")
    args = ap.parse_args()

    from gguf import GGUFReader

    reader = GGUFReader(str(args.gguf))
    hp = {k: gguf_scalar(reader, f"{ARCH}.{k}") for k in (
        "block_count", "embedding_length", "feed_forward_length", "attention.head_count",
        "attention.head_count_kv", "attention.key_length", "attention.layer_norm_rms_epsilon",
        "attention.sliding_window", "expert_count", "expert_used_count", "rope.freq_base",
        "rope.scaling.factor", "rope.scaling.original_context_length", "rope.scaling.yarn_beta_fast",
        "rope.scaling.yarn_beta_slow", "rope.scaling.yarn_truncate", "context_length", "embedding_length_out")}
    labels = gguf_labels(reader)
    n_layer, inter = int(hp["block_count"]), int(hp["feed_forward_length"])
    assert len(labels) == int(hp["embedding_length_out"]), "label count mismatch"
    print(f"gguf: {n_layer} layers, d_model {hp['embedding_length']}, inter {inter}, "
          f"{hp['expert_count']} experts top-{hp['expert_used_count']}, {len(labels)} labels")

    tensors = {t.name: t for t in reader.tensors}
    ref_header = read_ref_header(args.ref)
    ref_cfg = json.loads((args.ref / "config.json").read_text())

    # --- sanity: GGUF hparams vs the reference architecture config -----------
    checks = {
        "num_hidden_layers": n_layer, "hidden_size": int(hp["embedding_length"]), "intermediate_size": inter,
        "num_attention_heads": int(hp["attention.head_count"]), "num_key_value_heads": int(hp["attention.head_count_kv"]),
        "head_dim": int(hp["attention.key_length"]), "sliding_window": int(hp["attention.sliding_window"]),
        "num_local_experts": int(hp["expert_count"]), "num_experts_per_tok": int(hp["expert_used_count"]),
        "max_position_embeddings": int(hp["context_length"]),
    }
    for k, v in checks.items():
        if ref_cfg[k] != v:
            sys.exit(f"hparam mismatch {k}: gguf {v} vs ref {ref_cfg[k]}")
    rp = ref_cfg["rope_parameters"]
    for k, v in (("rope_theta", hp["rope.freq_base"]), ("factor", hp["rope.scaling.factor"]),
                 ("original_max_position_embeddings", hp["rope.scaling.original_context_length"]),
                 ("beta_fast", hp["rope.scaling.yarn_beta_fast"]), ("beta_slow", hp["rope.scaling.yarn_beta_slow"])):
        if abs(float(rp[k]) - float(v)) > 1e-6:
            sys.exit(f"rope mismatch {k}: gguf {v} vs ref {rp[k]}")
    if bool(rp.get("truncate", False)) != bool(hp["rope.scaling.yarn_truncate"]):
        sys.exit("rope truncate mismatch")
    if abs(float(ref_cfg["rms_norm_eps"]) - float(hp["attention.layer_norm_rms_epsilon"])) > 1e-9:
        sys.exit("rms_norm_eps mismatch")

    # --- build the HF tensor plan (name -> (producer, source gguf names)) ------
    # producer returns a float32/float16 numpy array in HF layout.
    plan: list[tuple[str, callable, list[str]]] = []

    def copy(name):
        return lambda: tensors[name].data

    plan.append(("model.embed_tokens.weight", copy("token_embd.weight"), ["token_embd.weight"]))
    for i in range(n_layer):
        p = f"blk.{i}."
        h = f"model.layers.{i}."
        simple = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
            "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
            "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
            "self_attn.o_proj.weight": "attn_output.weight", "self_attn.o_proj.bias": "attn_output.bias",
            "self_attn.sinks": "attn_sinks.weight",
            "post_attention_layernorm.weight": "post_attention_norm.weight",
            "mlp.router.weight": "ffn_gate_inp.weight", "mlp.router.bias": "ffn_gate_inp.bias",
            "mlp.experts.down_proj_bias": "ffn_down_exps.bias",
        }
        for hf_sub, g_sub in simple.items():
            plan.append((h + hf_sub, copy(p + g_sub), [p + g_sub]))

        def down(p=p):
            w = tensors[p + "ffn_down_exps.weight"].data          # [E, out, in]
            return np.ascontiguousarray(np.swapaxes(w, -1, -2))   # [E, in, out]

        def gate_up(p=p):
            g = tensors[p + "ffn_gate_exps.weight"].data          # [E, inter, in]
            u = tensors[p + "ffn_up_exps.weight"].data            # [E, inter, in]
            w = np.concatenate([g, u], axis=1)                    # [E, 2*inter, in]  (concatenated halves)
            return np.ascontiguousarray(np.swapaxes(w, -1, -2))   # [E, in, 2*inter]

        def gate_up_bias(p=p):
            g = tensors[p + "ffn_gate_exps.bias"].data            # [E, inter]
            u = tensors[p + "ffn_up_exps.bias"].data
            return np.ascontiguousarray(np.concatenate([g, u], axis=-1))

        plan.append((h + "mlp.experts.down_proj", down, [p + "ffn_down_exps.weight"]))
        plan.append((h + "mlp.experts.gate_up_proj", gate_up, [p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight"]))
        plan.append((h + "mlp.experts.gate_up_proj_bias", gate_up_bias, [p + "ffn_gate_exps.bias", p + "ffn_up_exps.bias"]))
    plan.append(("model.norm.weight", copy("output_norm.weight"), ["output_norm.weight"]))
    plan.append(("score.weight", copy("cls.output.weight"), ["cls.output.weight"]))
    plan.append(("score.bias", copy("cls.output.bias"), ["cls.output.bias"]))

    used = {g for _, _, gs in plan for g in gs}
    missing = sorted(used - set(tensors)); extra = sorted(set(tensors) - used)
    if missing or extra:
        sys.exit(f"tensor set mismatch: missing in gguf {missing}, unmapped gguf tensors {extra}")
    ref_names = {k for k in ref_header if k != "__metadata__"}
    plan_names = {n for n, _, _ in plan}
    if ref_names != plan_names:
        sys.exit(f"HF name set differs from reference: {sorted(ref_names ^ plan_names)}")

    # --- target dtype per tensor ----------------------------------------------
    def target_dtype(hf_name: str, src_dtype: np.dtype) -> str:
        ref_dt = ref_header[hf_name]["dtype"]
        if ref_dt == "F32":
            return "f32"                      # sinks
        assert ref_dt == "BF16", (hf_name, ref_dt)
        if src_dtype == np.float16:
            return args.weights_dtype         # lossy bf16 (default) or lossless f16
        return "bf16"                         # f32 source that is bf16-exact (checked below)

    # --- pass 1: shapes/dtypes -> safetensors header ---------------------------
    # Header order == data order (sorted by name), contiguous offsets.
    plan.sort(key=lambda x: x[0])
    records = []
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for hf_name, _, gnames in plan:
        g0 = tensors[gnames[0]]
        src_dt = g0.data.dtype
        if hf_name.endswith("mlp.experts.down_proj"):
            shape = [int(g0.data.shape[0]), int(g0.data.shape[2]), int(g0.data.shape[1])]
        elif hf_name.endswith("mlp.experts.gate_up_proj"):
            shape = [int(g0.data.shape[0]), int(g0.data.shape[2]), 2 * int(g0.data.shape[1])]
        elif hf_name.endswith("mlp.experts.gate_up_proj_bias"):
            shape = [int(g0.data.shape[0]), 2 * int(g0.data.shape[1])]
        else:
            shape = [int(s) for s in g0.data.shape]
        ref_shape = ref_header[hf_name]["shape"]
        if not hf_name.startswith("score.") and shape != ref_shape:
            sys.exit(f"shape mismatch {hf_name}: {shape} vs reference {ref_shape}")
        tdt = target_dtype(hf_name, src_dt)
        nbytes = int(np.prod(shape)) * (4 if tdt == "f32" else 2)
        header[hf_name] = {"dtype": ST_DTYPE[tdt], "shape": shape, "data_offsets": [offset, offset + nbytes]}
        records.append({"name": hf_name, "shape": shape, "dtype": ST_DTYPE[tdt], "source": gnames,
                        "source_dtype": str(src_dt), "lossless": None, "n_f16_subnormal": None})
        offset += nbytes
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr += b" " * ((8 - len(hdr) % 8) % 8)

    # --- pass 2: stream tensors -----------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    st_path = args.out / "model.safetensors"
    n_lossy = 0
    n_subnormal_total = 0
    with open(st_path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for (hf_name, produce, _), rec in zip(plan, records):
            arr = produce()
            if not np.all(np.isfinite(arr.astype(np.float32, copy=False))):
                sys.exit(f"non-finite values in {hf_name}")
            tdt = rec["dtype"]
            if arr.dtype == np.float16:
                a = np.abs(arr.astype(np.float32))
                rec["n_f16_subnormal"] = int(np.count_nonzero((a > 0) & (a < 6.1035156e-5)))
                n_subnormal_total += rec["n_f16_subnormal"]
            if tdt == "F32":
                out = np.ascontiguousarray(arr, dtype=np.float32)
                rec["lossless"] = True
            elif tdt == "F16":
                assert arr.dtype == np.float16
                out = np.ascontiguousarray(arr)
                rec["lossless"] = True
            else:  # BF16
                a32 = arr.astype(np.float32)
                exact = is_bf16_exact(a32)
                if arr.dtype == np.float32 and not exact:
                    sys.exit(f"{hf_name}: F32 gguf tensor is not bf16-exact; refusing silent loss (use a f32 target)")
                rec["lossless"] = bool(exact)
                if not exact:
                    n_lossy += 1
                out = f32_to_bf16_bits(a32)
            b = out.tobytes()
            assert len(b) == header[hf_name]["data_offsets"][1] - header[hf_name]["data_offsets"][0], hf_name
            f.write(b)
            print(f"  {hf_name:55s} {str(rec['shape']):20s} {rec['source_dtype']:>8s} -> {tdt:4s}"
                  f"{'' if rec['lossless'] else '  (lossy rounding)'}")
    print(f"wrote {st_path} ({offset/1e9:.3f} GB, {len(records)} tensors, {n_lossy} tensors re-rounded to bf16, "
          f"{n_subnormal_total} f16-subnormal elements)")

    # --- config.json -----------------------------------------------------------
    cfg = dict(ref_cfg)
    cfg.pop("transformers.js_config", None)
    cfg["architectures"] = ["OpenAIPrivacyFilterForTokenClassification"]
    cfg["model_type"] = "openai_privacy_filter"
    cfg["id2label"] = {str(i): l for i, l in enumerate(labels)}
    cfg["label2id"] = {l: i for i, l in enumerate(labels)}
    cfg["dtype"] = CFG_DTYPE[args.weights_dtype]
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- tokenizer (o200k; identical vocab/merges to the GGUF's tokenizer.ggml.*) --
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(args.ref / fn, args.out / fn)
    # cross-check vocab size against the GGUF token table
    n_gguf_tokens = len(reader.fields["tokenizer.ggml.tokens"].data)
    if n_gguf_tokens != cfg["vocab_size"]:
        sys.exit(f"gguf tokenizer has {n_gguf_tokens} tokens, config vocab_size {cfg['vocab_size']}")

    # --- manifest --------------------------------------------------------------
    gguf_sha = None if args.skip_gguf_sha else sha256_file(args.gguf)
    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "converter": "toramon/scripts/convert_gguf_to_hf.py",
        "source_gguf": str(args.gguf),
        "source_gguf_sha256": gguf_sha,
        "source_gguf_sha256_expected": EXPECTED_GGUF_SHA256,
        "source_gguf_sha256_matches_expected": (gguf_sha == EXPECTED_GGUF_SHA256) if gguf_sha else None,
        "source_gguf_general_name": gguf_scalar(reader, "general.name"),
        "reference_checkpoint": str(args.ref),
        "weights_dtype": args.weights_dtype,
        "model_safetensors_sha256": sha256_file(st_path),
        "model_safetensors_bytes": st_path.stat().st_size,
        "n_tensors": len(records),
        "n_tensors_lossy_f16_to_bf16": n_lossy,
        "n_f16_subnormal_elements": n_subnormal_total,
        "n_labels": len(labels),
        "labels": labels,
        "notes": [
            "Reconstruction from the author's F16 GGUF, not the author's original checkpoint.",
            "F16 -> BF16 is round-to-nearest-even; 'lossless' per tensor means every element was already bf16-exact "
            "(n_tensors_lossy_f16_to_bf16 == 0 means the whole file is a lossless re-encoding of the GGUF).",
            "Elements with 0 < |x| < 6.1e-5 (n_f16_subnormal_elements) sat on the f16 subnormal grid in the GGUF and may "
            "differ from the author's original bf16 values; this cannot be recovered from the GGUF.",
            "GGUF F32 tensors (norms, biases, router) verified bf16-exact and stored as bf16 losslessly; sinks kept f32.",
            "gate_up_proj rebuilt from CONCATENATED gate|up halves; down_proj/gate_up_proj transposed (inverse of convert.py).",
            "tokenizer.json / tokenizer_config.json copied from the OpenAI reference (same o200k vocab as the GGUF).",
        ],
        "tensors": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"model.safetensors sha256 {manifest['model_safetensors_sha256']}")
    if gguf_sha:
        print(f"source gguf sha256 {gguf_sha} (expected match: {manifest['source_gguf_sha256_matches_expected']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
