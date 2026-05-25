# anima_lora_hierarchy_cut.py
# Pure-stdlib safetensors LoRA hierarchy cutter for Anima / KModel-style LoRAs.
#
# Features:
# - Drag & drop one or more .safetensors LoRA files onto the companion .bat file.
# - Also works from the command line:
#     python anima_lora_hierarchy_cut.py path\to\lora.safetensors
# - Reads and repacks safetensors files without torch, numpy, or the safetensors package.
# - Keeps the source LoRA unchanged.
# - Writes generated LoRAs to the same folder as the source LoRA by default.
# - Avoids double underscores in generated filenames because sd-dynamic-prompts treats
#   __name__ as wildcard syntax.
#
# Target LoRA key style:
# - Anima / KModel-style LoRA keys such as:
#     lora_unet_blocks_0_...
#     lora_unet_blocks_27_...
#     lora_te_layers_0_...
#
# Notes:
# - For SDXL / Illustrious / A1111-style LoRAs using input_blocks / middle_block /
#   output_blocks naming, this script may not cut the intended layers.

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


SAFE_SEPARATOR = "_CUT_"
COPY_BUFFER_SIZE = 1024 * 1024 * 16

LORA_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".alpha",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
    ".diff",
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
)


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: Tuple[int, ...]
    start_abs: int
    end_abs: int


@dataclass(frozen=True)
class Variant:
    suffix: str
    description: str
    should_cut_module: Callable[[str], bool]


def read_safetensors_header(path: Path) -> Tuple[Dict[str, Any], int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Invalid safetensors file, cannot read header length: {path}")

        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Invalid safetensors file, cannot read full header: {path}")

    header = json.loads(header_bytes.decode("utf-8"))
    return header, 8 + header_len


def parse_tensors(path: Path) -> Tuple[List[TensorInfo], Dict[str, str] | None]:
    header, data_start_abs = read_safetensors_header(path)
    metadata = header.get("__metadata__")

    tensors: List[TensorInfo] = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            continue

        offsets = value.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Tensor has invalid data_offsets: {key}")

        start_rel, end_rel = int(offsets[0]), int(offsets[1])
        tensors.append(
            TensorInfo(
                key=key,
                dtype=str(value.get("dtype", "UNKNOWN")),
                shape=tuple(int(x) for x in value.get("shape", [])),
                start_abs=data_start_abs + start_rel,
                end_abs=data_start_abs + end_rel,
            )
        )

    return tensors, metadata


def lora_module_name(key: str) -> str:
    for suffix in sorted(LORA_SUFFIXES, key=len, reverse=True):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def is_text_encoder_module(module: str) -> bool:
    m = module.lower()
    return (
        m.startswith("lora_te_")
        or m.startswith("lora_te1_")
        or m.startswith("lora_te2_")
        or m.startswith("lora_text_encoder")
    )


def is_anima_dit_module(module: str) -> bool:
    return module.lower().startswith("lora_unet_blocks_")


def get_anima_dit_block_index(module: str) -> int | None:
    match = re.match(r"lora_unet_blocks_(\d+)_", module.lower())
    return int(match.group(1)) if match else None


def is_anima_dit_block_in(module: str, start: int, end: int) -> bool:
    idx = get_anima_dit_block_index(module)
    return idx is not None and start <= idx <= end


def is_anima_dit_mlp_module(module: str) -> bool:
    m = module.lower()
    return m.startswith("lora_unet_blocks_") and "_mlp_" in m


def is_anima_dit_cross_attn_module(module: str) -> bool:
    m = module.lower()
    return m.startswith("lora_unet_blocks_") and "_cross_attn_" in m


def is_anima_dit_self_attn_module(module: str) -> bool:
    m = module.lower()
    return m.startswith("lora_unet_blocks_") and "_self_attn_" in m


def is_anima_dit_late_mlp_19_27(module: str) -> bool:
    return is_anima_dit_mlp_module(module) and is_anima_dit_block_in(module, 19, 27)


def sanitize_name_part(text: str) -> str:
    # Avoid sd-dynamic-prompts wildcard syntax.
    while "__" in text:
        text = text.replace("__", "_")

    # Keep names reasonably filesystem-safe.
    for ch in '<>:"/\\|?*':
        text = text.replace(ch, "_")

    return text


def output_name_for(src_path: Path, suffix: str) -> str:
    safe_stem = sanitize_name_part(src_path.stem)
    safe_suffix = sanitize_name_part(suffix)
    name = f"{safe_stem}{SAFE_SEPARATOR}{safe_suffix}.safetensors"

    # Final guard.
    while "__" in name:
        name = name.replace("__", "_")

    return name


def build_new_header(kept: List[TensorInfo], metadata: Dict[str, str] | None) -> Tuple[bytes, List[TensorInfo]]:
    new_header: Dict[str, Any] = {}
    if metadata:
        new_header["__metadata__"] = metadata

    offset = 0
    for tensor in kept:
        byte_len = tensor.end_abs - tensor.start_abs
        new_header[tensor.key] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + byte_len],
        }
        offset += byte_len

    header_bytes = json.dumps(new_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # Padding is not strictly required by the safetensors spec, but keeps the header neat.
    pad_len = (8 - (len(header_bytes) % 8)) % 8
    if pad_len:
        header_bytes += b" " * pad_len

    return header_bytes, kept


def copy_range(src_file, dst_file, start: int, end: int) -> None:
    remaining = end - start
    src_file.seek(start)

    while remaining > 0:
        chunk = src_file.read(min(COPY_BUFFER_SIZE, remaining))
        if not chunk:
            raise IOError("Unexpected EOF while copying tensor bytes")
        dst_file.write(chunk)
        remaining -= len(chunk)


def write_variant(
    src_path: Path,
    out_path: Path,
    tensors: List[TensorInfo],
    metadata: Dict[str, str] | None,
    variant: Variant,
    overwrite: bool,
) -> Dict[str, int | str]:
    modules_before = {lora_module_name(t.key) for t in tensors}
    cut_modules = {m for m in modules_before if variant.should_cut_module(m)}

    kept = [t for t in tensors if lora_module_name(t.key) not in cut_modules]
    cut = [t for t in tensors if lora_module_name(t.key) in cut_modules]

    if not cut:
        return {
            "status": "skipped_no_matching_modules",
            "kept_tensors": len(kept),
            "cut_tensors": 0,
            "kept_modules": len(modules_before),
            "cut_modules": 0,
        }

    if not kept:
        raise ValueError(f"Variant would remove every tensor: {variant.suffix}")

    header_bytes, copy_plan = build_new_header(kept, metadata)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with src_path.open("rb") as src, tmp_path.open("wb") as dst:
        dst.write(struct.pack("<Q", len(header_bytes)))
        dst.write(header_bytes)

        for tensor in copy_plan:
            copy_range(src, dst, tensor.start_abs, tensor.end_abs)

    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            tmp_path.unlink(missing_ok=True)
            return {
                "status": "skipped_exists",
                "kept_tensors": len(kept),
                "cut_tensors": len(cut),
                "kept_modules": len({lora_module_name(t.key) for t in kept}),
                "cut_modules": len(cut_modules),
            }

    tmp_path.replace(out_path)

    return {
        "status": "written",
        "kept_tensors": len(kept),
        "cut_tensors": len(cut),
        "kept_modules": len({lora_module_name(t.key) for t in kept}),
        "cut_modules": len(cut_modules),
    }


def get_variants() -> List[Variant]:
    return [
        Variant(
            suffix="STYLE_RECOMMENDED_TEcut",
            description="Style LoRA first choice: remove Text Encoder LoRA only; keep all DiT-side modules.",
            should_cut_module=lambda m: is_text_encoder_module(m),
        ),
        Variant(
            suffix="CHARACTER_RECOMMENDED_no_late_MLP_19_27",
            description="Character LoRA first choice: keep Text Encoder and attention; remove only DiT MLP modules in blocks 19-27.",
            should_cut_module=lambda m: is_anima_dit_late_mlp_19_27(m),
        ),
        Variant(
            suffix="CHARACTER_STRONG_no_MLP",
            description="Character LoRA stronger style reduction: keep Text Encoder and attention; remove all DiT MLP modules.",
            should_cut_module=lambda m: is_anima_dit_mlp_module(m),
        ),
        Variant(
            suffix="STYLE_LIGHT_TEcut_no_MLP",
            description="Style-light diagnostic: remove Text Encoder and all DiT MLP modules; keep DiT attention only.",
            should_cut_module=lambda m: is_text_encoder_module(m) or is_anima_dit_mlp_module(m),
        ),
        Variant(
            suffix="DIAGNOSTIC_no_late_blocks_19_27",
            description="Diagnostic only: remove full late DiT blocks 19-27. This may damage character reproduction.",
            should_cut_module=lambda m: is_anima_dit_block_in(m, 19, 27),
        ),
        Variant(
            suffix="DIAGNOSTIC_TEcut_no_late_blocks_19_27",
            description="Diagnostic only: remove Text Encoder and full late DiT blocks 19-27.",
            should_cut_module=lambda m: is_text_encoder_module(m) or is_anima_dit_block_in(m, 19, 27),
        ),
        Variant(
            suffix="DIAGNOSTIC_no_cross_attn",
            description="Diagnostic only: remove DiT cross-attention modules.",
            should_cut_module=lambda m: is_anima_dit_cross_attn_module(m),
        ),
        Variant(
            suffix="DIAGNOSTIC_no_self_attn",
            description="Diagnostic only: remove DiT self-attention modules.",
            should_cut_module=lambda m: is_anima_dit_self_attn_module(m),
        ),
    ]


def summarize_modules(tensors: List[TensorInfo]) -> Dict[str, Any]:
    modules = sorted({lora_module_name(t.key) for t in tensors})
    te_modules = [m for m in modules if is_text_encoder_module(m)]
    dit_modules = [m for m in modules if is_anima_dit_module(m)]
    mlp_modules = [m for m in modules if is_anima_dit_mlp_module(m)]
    cross_modules = [m for m in modules if is_anima_dit_cross_attn_module(m)]
    self_modules = [m for m in modules if is_anima_dit_self_attn_module(m)]

    block_counts: Dict[int, int] = {}
    for module in dit_modules:
        idx = get_anima_dit_block_index(module)
        if idx is not None:
            block_counts[idx] = block_counts.get(idx, 0) + 1

    return {
        "modules": modules,
        "te_modules": te_modules,
        "dit_modules": dit_modules,
        "mlp_modules": mlp_modules,
        "cross_modules": cross_modules,
        "self_modules": self_modules,
        "block_counts": block_counts,
    }


def write_manifest(src_path: Path, output_dir: Path, tensors: List[TensorInfo], results) -> Path:
    manifest_name = sanitize_name_part(f"{src_path.stem}{SAFE_SEPARATOR}hierarchy_cut_manifest.txt")
    manifest_path = output_dir / manifest_name

    summary = summarize_modules(tensors)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("Anima LoRA hierarchy cut manifest\n")
        f.write("Generated by anima_lora_hierarchy_cut.py\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source LoRA: {src_path}\n")
        f.write(f"Output dir : {output_dir}\n\n")

        f.write("Original summary\n")
        f.write("----------------\n")
        f.write(f"Tensor count              : {len(tensors)}\n")
        f.write(f"Module count              : {len(summary['modules'])}\n")
        f.write(f"Text Encoder modules      : {len(summary['te_modules'])}\n")
        f.write(f"DiT modules               : {len(summary['dit_modules'])}\n")
        f.write(f"DiT MLP modules           : {len(summary['mlp_modules'])}\n")
        f.write(f"DiT cross-attn modules    : {len(summary['cross_modules'])}\n")
        f.write(f"DiT self-attn modules     : {len(summary['self_modules'])}\n")

        f.write("\nDiT block module counts:\n")
        if summary["block_counts"]:
            for idx in sorted(summary["block_counts"]):
                f.write(f"  blocks {idx:02d}: {summary['block_counts'][idx]}\n")
        else:
            f.write("  <no lora_unet_blocks_N modules found; this may not be an Anima/KModel LoRA>\n")

        f.write("\nGenerated variants\n")
        f.write("------------------\n")
        for variant, out_path, stats in results:
            f.write(f"\n[{variant.suffix}]\n")
            f.write(f"Description : {variant.description}\n")
            f.write(f"Output      : {out_path}\n")
            for k, v in stats.items():
                f.write(f"{k:17}: {v}\n")

        f.write("\nSuggested first tests\n")
        f.write("---------------------\n")
        f.write("Style LoRA first choice:\n")
        f.write(f"  {output_name_for(src_path, 'STYLE_RECOMMENDED_TEcut')}\n")
        f.write("Character LoRA first choice:\n")
        f.write(f"  {output_name_for(src_path, 'CHARACTER_RECOMMENDED_no_late_MLP_19_27')}\n")
        f.write("Character LoRA stronger style reduction:\n")
        f.write(f"  {output_name_for(src_path, 'CHARACTER_STRONG_no_MLP')}\n")

    return manifest_path


def process_one(src_path: Path, output_dir: Path | None, overwrite: bool, selected_variants: set[str] | None) -> int:
    print("")
    print("=" * 100)
    print(f"Processing LoRA: {src_path}")
    print("=" * 100)

    if not src_path.exists():
        print(f"ERROR: file not found: {src_path}")
        return 1

    if src_path.suffix.lower() != ".safetensors":
        print(f"SKIP: not a .safetensors file: {src_path}")
        return 1

    out_dir = output_dir if output_dir is not None else src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading safetensors header...")
    tensors, metadata = parse_tensors(src_path)

    summary = summarize_modules(tensors)
    print(f"Original tensors       : {len(tensors)}")
    print(f"Original modules       : {len(summary['modules'])}")
    print(f"Text Encoder modules   : {len(summary['te_modules'])}")
    print(f"DiT modules            : {len(summary['dit_modules'])}")
    print(f"DiT block count        : {len(summary['block_counts'])}")
    if not summary["block_counts"]:
        print("WARNING: no lora_unet_blocks_N modules found. This may not be an Anima/KModel LoRA.")
    print("")

    variants = get_variants()
    if selected_variants:
        variants = [v for v in variants if v.suffix in selected_variants]

    if not variants:
        print("ERROR: no matching variants selected.")
        return 1

    results = []
    for variant in variants:
        out_path = out_dir / output_name_for(src_path, variant.suffix)
        print(f"Creating: {out_path.name}")
        print(f"  {variant.description}")
        stats = write_variant(src_path, out_path, tensors, metadata, variant, overwrite=overwrite)
        print(
            f"  status={stats['status']}, "
            f"kept_modules={stats['kept_modules']}, cut_modules={stats['cut_modules']}, "
            f"kept_tensors={stats['kept_tensors']}, cut_tensors={stats['cut_tensors']}"
        )
        print("")
        results.append((variant, out_path, stats))

    manifest_path = write_manifest(src_path, out_dir, tensors, results)
    print(f"Manifest: {manifest_path}")
    print("Done.")

    print("")
    print("Recommended first files:")
    print(f"  Style LoRA              : {output_name_for(src_path, 'STYLE_RECOMMENDED_TEcut')}")
    print(f"  Character LoRA          : {output_name_for(src_path, 'CHARACTER_RECOMMENDED_no_late_MLP_19_27')}")
    print(f"  Character stronger cut  : {output_name_for(src_path, 'CHARACTER_STRONG_no_MLP')}")
    return 0


def prompt_for_paths() -> List[Path]:
    print("No LoRA file was supplied.")
    print("Drag & drop a .safetensors file onto the .bat file, or paste a full path here.")
    print("Press Enter without input to exit.")
    text = input("> ").strip().strip('"')
    if not text:
        return []
    return [Path(text)]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut selected hierarchy/modules from Anima / KModel-style LoRA safetensors files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more LoRA .safetensors files. Drag & drop also works through the companion .bat file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: same folder as each source LoRA.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files if they already exist.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Generate only selected variant suffixes. Use --list-variants to see names.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variant suffixes and exit.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list_variants:
        print("Available variants:")
        for variant in get_variants():
            print(f"- {variant.suffix}")
            print(f"  {variant.description}")
        return 0

    print("Anima LoRA hierarchy cutter")
    print("===========================")
    print("No torch / numpy / safetensors package required.")
    print("Generated filenames avoid double underscores for sd-dynamic-prompts compatibility.")
    print("")

    paths = [Path(p.strip().strip('"')) for p in args.files]
    if not paths:
        paths = prompt_for_paths()

    if not paths:
        print("No input. Exit.")
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else None
    selected_variants = set(args.only) if args.only else None

    exit_code = 0
    for path in paths:
        try:
            code = process_one(path, output_dir, overwrite=args.overwrite, selected_variants=selected_variants)
            exit_code = max(exit_code, code)
        except Exception as e:
            print("")
            print(f"ERROR while processing: {path}")
            print(e)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
