# Anima LoRA Hierarchy Cutter

A small pure-Python utility for cutting selected hierarchy/modules from Anima / KModel-style LoRA `.safetensors` files.

It was made for workflows where you want to separate character/style influence in Anima-style LoRAs, especially when combining character LoRAs with separate style LoRAs.

## Features

- No `torch`, `numpy`, or `safetensors` Python package required.
- Reads and repacks `.safetensors` files using only Python standard library.
- Keeps the original LoRA unchanged.
- Supports drag-and-drop through the included Windows batch file.
- Supports command-line usage.
- Generated filenames avoid double underscores (`__`) because `sd-dynamic-prompts` treats `__name__` as wildcard syntax.

## Target LoRA format

This script is intended for Anima / KModel-style LoRA keys such as:

```text
lora_unet_blocks_0_...
lora_unet_blocks_27_...
lora_te_layers_0_...
```

For SDXL / Illustrious / A1111-style LoRAs using `input_blocks`, `middle_block`, and `output_blocks`, the script may not cut the intended layers.

## Usage

### Drag & drop

Place these two files in the same folder:

```text
anima_lora_hierarchy_cut.py
run_anima_lora_hierarchy_cut.bat
```

Then drag one or more `.safetensors` LoRA files onto:

```text
run_anima_lora_hierarchy_cut.bat
```

Generated files are created in the same folder as the source LoRA.

### Command line

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors"
```

List available variants:

```bat
python anima_lora_hierarchy_cut.py --list-variants
```

Generate only one variant:

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --only CHARACTER_RECOMMENDED_no_late_MLP_19_27
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

Cuts Text Encoder LoRA modules and keeps DiT-side modules.

Use this when a style LoRA affects prompt interpretation too much and you want to keep mainly the visual style side.

### Character LoRA first choice

```text
*_CUT_CHARACTER_RECOMMENDED_no_late_MLP_19_27.safetensors
```

Keeps Text Encoder and attention modules, and cuts only DiT MLP modules in late blocks 19-27.

This is a balanced first choice for character LoRAs when you want to keep character reproduction while reducing some style/finish influence.

### Stronger character style reduction

```text
*_CUT_CHARACTER_STRONG_no_MLP.safetensors
```

Keeps Text Encoder and attention modules, and cuts all DiT MLP modules.

This can reduce style/texture influence more strongly, while often preserving character trigger behavior.

## Notes

This script uses deletion-based cutting. It does not rescale LoRA weights; it removes selected tensor keys and repacks the safetensors file.

A manifest file is generated next to the output LoRAs to show how many modules/tensors were kept or cut.
