# Depth Anything 3 自前Q4版

ByteDance Seedの公式DA3-BASEとDA3-LARGE-1.1の重みを使い、再学習せずに単画像の相対奥行きだけを出力するONNX（opset 17）に変換した。504×504のRGB入力に対して、定数の行列積の重みをブロック幅32の符号付き4ビットに量子化する。縦横比はアプリ側で余白を付けて維持し、点群化の前に取り除く。

| 公式モデルと固定リビジョン | ONNX容量 | ONNX SHA-256 |
|---|---:|---|
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE/tree/f4a6c9b3c95e41c82048423d3493a81ec3fa810e) | 107,009,016 B | `f93ef38159a3a76fa6de2513c0170e4a563044ee4c39c70d24805390cf4b69f9` |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1/tree/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c) | 329,828,526 B | `216100b23bbaeca49df8b18b0fad18a98349402a3c4bb71b1a3d094eb4cc63c0` |

## 再変換

各公式リポジトリから `config.json` と `model.safetensors` を上表のモデル名のディレクトリに取得する。[公式実装](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) の `src` を `PYTHONPATH` に加え、以下を実行する。変換時には元の重みのSHA-256を検査する。

```sh
python scripts/export_da3.py DA3-BASE da3-base-q4.onnx
python scripts/export_da3.py DA3-LARGE-1.1 da3-large-1.1-q4.onnx
```

検証環境はPython 3.10、torch 2.7.1、onnx 1.17.0、onnxruntime 1.22.0、numpy 1.26.4、safetensors 0.5.3、einops 0.8.1、omegaconf 2.3.0、addict 2.4.0。

## 検証結果

固定実写6枚（T0503、T0506、T0508、T0512、T0517、T0519）とT0512の縦長切り出し1件で、公式PyTorchの相対奥行きとPC版EdgeのWebGPU出力を比較した。縦長では余白を評価から除いた。

| モデル | 画素ごとの相関係数の範囲 | 平均絶対差 ÷ 公式出力の平均 |
|---|---:|---:|
| DA3-BASE Q4 | 0.9980–0.9995 | 2.00–8.16% |
| DA3-LARGE-1.1 Q4 | 0.9978–0.9999 | 1.28–5.13% |

この数値は量子化前後の出力差を示す。実測距離との誤差は未評価。両モデルともONNX Runtime Web 1.22.0の単一スレッドWASMでも正方形と縦長の入力で有限値が得られた。

## 配布

[GitHub Release](https://github.com/kasys1422/cone-vision/releases/tag/depth-da3-q4-v1)に個別のZIPを置く。Pagesビルドは両ZIPを取得してSHA-256を検証する。先頭の `model.onnx` は非圧縮エントリで、ブラウザーではダウンロード後にその領域を切り出し、モデル本体のSHA-256を照合して保存する。
