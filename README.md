# Anima LoRA Hierarchy Cutter

[日本語版はこちら](README.ja.md)

A small standalone utility for cutting selected hierarchy/modules from Anima / KModel-style LoRA `.safetensors` files.

The tool is intended for experiments such as:

- reducing style influence from character LoRAs
- reducing prompt-conditioning influence from style LoRAs
- creating lighter diagnostic variants of an existing LoRA
- comparing how Text Encoder, attention, and MLP modules affect the result

The original LoRA file is never overwritten.

## Features

- Works by drag-and-drop on Windows
- Also works from the command line
- Uses only the Python standard library
- Does not require `torch`, `numpy`, or the `safetensors` package
- Reads only the safetensors header and tensor byte ranges
- Re-packs selected tensors into new `.safetensors` files
- Writes output files next to the source LoRA by default

## Supported LoRA structure

This script is intended for Anima / KModel-style LoRAs with keys like:

```text
lora_unet_blocks_0_...
lora_unet_blocks_27_...
lora_te_layers_0_...
```

It is not designed for SDXL / Illustrious / A1111-style LoRAs that use structures such as:

```text
lora_unet_input_blocks_...
lora_unet_middle_block_...
lora_unet_output_blocks_...
```

Those files may run through the script, but the intended hierarchy cuts may not be applied.

## Files

```text
anima_lora_hierarchy_cut.py
run_anima_lora_hierarchy_cut.bat
README.md
README.ja.md
```

## Usage

### Drag and drop

Place the Python file and batch file in the same folder.

```text
anima_lora_hierarchy_cut.py
run_anima_lora_hierarchy_cut.bat
```

Then drag one or more LoRA `.safetensors` files onto:

```text
run_anima_lora_hierarchy_cut.bat
```

Generated files will be created in the same folder as the source LoRA.

### Command line

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors"
```

List available cut variants:

```bat
python anima_lora_hierarchy_cut.py --list-variants
```

Generate only one variant:

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --only CHARACTER_RECOMMENDED_no_late_MLP_19_27
```

Write outputs to a specific folder:

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --output-dir "C:\path\to\output"
```

Overwrite existing generated files:

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --overwrite
```

## Main generated variants

### Style LoRA first choice

```text
*_CUT_STYLE_RECOMMENDED_TEcut.safetensors
```

Removes Text Encoder LoRA modules and keeps DiT-side modules.

This is useful when you want a style LoRA to affect mainly the visual rendering side while reducing its effect on text conditioning.

### Character LoRA first choice

```text
*_CUT_CHARACTER_RECOMMENDED_no_late_MLP_19_27.safetensors
```

Keeps Text Encoder and attention modules, and removes only DiT MLP modules in late blocks 19-27.

This is a balanced first choice for character LoRAs when you want to keep character reproduction while reducing some style or finishing influence.

### Stronger character style reduction

```text
*_CUT_CHARACTER_STRONG_no_MLP.safetensors
```

Keeps Text Encoder and attention modules, and removes all DiT MLP modules.

This can reduce style or texture influence more strongly, while often preserving character trigger behavior.

### Diagnostic variants

The script also generates diagnostic variants such as:

```text
*_CUT_DIAGNOSTIC_no_late_blocks_19_27.safetensors
*_CUT_DIAGNOSTIC_TEcut_no_late_blocks_19_27.safetensors
*_CUT_DIAGNOSTIC_no_cross_attn.safetensors
*_CUT_DIAGNOSTIC_no_self_attn.safetensors
```

These are mainly for testing how each module group affects the result.

## How it works

This is a deletion-based cutter.

It does not multiply or rescale LoRA weights. Instead, it removes selected tensor keys and writes a new safetensors file containing only the remaining tensors.

A manifest file is generated next to the output files. It records:

- source LoRA path
- output file paths
- tensor counts
- module counts
- kept/cut module counts for each variant

## Notes

- The output files are experimental variants. Keep the original LoRA.
- Whether a variant is useful depends on how the LoRA was trained.
- For character LoRAs, removing too much of the late DiT blocks may reduce character reproduction.
- For style LoRAs, removing Text Encoder modules is often a useful first test.
