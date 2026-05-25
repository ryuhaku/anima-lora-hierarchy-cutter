# Anima LoRA Hierarchy Cutter

[English version](README.md)

Anima / KModel 形式の LoRA `.safetensors` に対して、特定の階層・モジュールを削除した派生LoRAを作成するための小さなスタンドアロンツールです。

主に以下のような実験を想定しています。

- キャラLoRAから画風寄りの影響を弱める
- 画風LoRAからテキスト条件側への影響を弱める
- 既存LoRAの軽量・診断用バリエーションを作る
- Text Encoder、attention、MLP のどこが出力に効いているかを比較する

元のLoRAファイルは上書きしません。

## 特徴

- Windowsでドラッグ&ドロップ実行可能
- コマンドライン実行にも対応
- Python標準ライブラリのみで動作
- `torch`、`numpy`、`safetensors` パッケージ不要
- safetensorsのヘッダとtensor byte rangeのみを読み取り
- 選択したtensorだけを再パックして新しい `.safetensors` を作成
- デフォルトでは元LoRAと同じフォルダに出力

## 対応しているLoRA構造

このスクリプトは、以下のようなキー構造を持つ Anima / KModel 形式のLoRAを想定しています。

```text
lora_unet_blocks_0_...
lora_unet_blocks_27_...
lora_te_layers_0_...
```

一方で、SDXL / Illustrious / A1111系でよく使われる以下のような構造は対象外です。

```text
lora_unet_input_blocks_...
lora_unet_middle_block_...
lora_unet_output_blocks_...
```

対象外のLoRAでも実行自体はできる場合がありますが、意図した階層カットにはならない可能性があります。

## ファイル構成

```text
anima_lora_hierarchy_cut.py
run_anima_lora_hierarchy_cut.bat
README.md
README.ja.md
```

## 使い方

### ドラッグ&ドロップ

以下2ファイルを同じフォルダに置きます。

```text
anima_lora_hierarchy_cut.py
run_anima_lora_hierarchy_cut.bat
```

その後、対象のLoRA `.safetensors` ファイルを以下のbatにドラッグ&ドロップします。

```text
run_anima_lora_hierarchy_cut.bat
```

生成ファイルは、元LoRAと同じフォルダに作成されます。

### コマンドライン

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors"
```

生成可能なバリエーション一覧を表示します。

```bat
python anima_lora_hierarchy_cut.py --list-variants
```

特定のバリエーションだけ生成します。

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --only CHARACTER_RECOMMENDED_no_late_MLP_19_27
```

出力先を指定します。

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --output-dir "C:\path\to\output"
```

既存の生成ファイルを上書きします。

```bat
python anima_lora_hierarchy_cut.py "C:\path\to\your_lora.safetensors" --overwrite
```

## 主な生成バリエーション

### 画風LoRA向けの第一候補

```text
*_CUT_STYLE_RECOMMENDED_TEcut.safetensors
```

Text Encoder側のLoRAモジュールを削除し、DiT側のモジュールを残します。

画風LoRAを主に描画・画風側へ効かせつつ、テキスト条件側への影響を弱めたい場合の第一候補です。

### キャラLoRA向けの第一候補

```text
*_CUT_CHARACTER_RECOMMENDED_no_late_MLP_19_27.safetensors
```

Text Encoderとattention系モジュールを残し、後段 DiT block 19〜27 の MLP だけを削除します。

キャラ再現を残しつつ、画風・仕上げ寄りの影響を少し弱めたい場合のバランス型です。

### キャラLoRAの画風影響を強めに落とす候補

```text
*_CUT_CHARACTER_STRONG_no_MLP.safetensors
```

Text Encoderとattention系モジュールを残し、DiT側のMLPをすべて削除します。

画風・質感への影響をより強く弱めたい場合の候補です。トリガーワードによるキャラ再現は残る場合がありますが、LoRAによって結果は変わります。

### 診断用バリエーション

以下のような診断用ファイルも生成されます。

```text
*_CUT_DIAGNOSTIC_no_late_blocks_19_27.safetensors
*_CUT_DIAGNOSTIC_TEcut_no_late_blocks_19_27.safetensors
*_CUT_DIAGNOSTIC_no_cross_attn.safetensors
*_CUT_DIAGNOSTIC_no_self_attn.safetensors
```

各モジュール群が出力にどう影響しているかを見るための比較用です。

## 仕組み

このツールは削除型の階層カットを行います。

LoRAの重みを0.5倍などにスケーリングするのではなく、対象となるtensor keyを削除し、残ったtensorだけで新しいsafetensorsファイルを作成します。

出力ファイルと同じ場所にmanifestファイルも作成され、以下を確認できます。

- 元LoRAのパス
- 出力ファイルのパス
- tensor数
- module数
- 各バリエーションで残した/削除したmodule数

## 注意事項

- 生成ファイルは実験用の派生LoRAです。元LoRAは必ず残してください。
- どのバリエーションが有効かはLoRAの学習内容によって変わります。
- キャラLoRAでは、後段DiT blockを大きく削るとキャラ再現が弱くなる場合があります。
- 画風LoRAでは、Text Encoder側を削ることが有効な初手になる場合があります。
