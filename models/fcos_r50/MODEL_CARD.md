# FCOS ResNet-50 FPN（512px）

実写コーン用の単体検出器。写真をアスペクト比を保って512×512へ縮小・余白補完し、5段の特徴ピラミッドから位置とスコアを予測する。検出後の5点推定にはV12の192pxモデルを使う。

## 固定実写テスト

TraConの43枚・63個、うち空画像21枚。Web配布用ONNXをONNX Runtime Web 1.22.0の単一スレッドWASMとOpenCV.jsで実行。両構成でしきい値0.50、NMSと正解対応のIoUは0.50。開発で以前から使用しているテストセットであり、新しい撮影現場での評価は別途必要。

| 指標 | V10 YOLOX-Nano | FCOS ResNet-50 |
|---|---:|---:|
| 正検出 / 誤検出 / 見落とし | 42 / 9 / 21 | 55 / 3 / 8 |
| 正解枠の高さ96px未満 | 0/7 | 2/7 |
| 空画像での誤検出枠 | 0 | 0 |
| 未検出を含む可視点PCK@0.05 | 37/63点 | 44/63点 |
| 推論時間の中央値（PC、モデル読込を除く） | 38 ms | 2,136 ms |

T0020では左と中央を検出し、黒い支柱は除外。右端で一部だけ写った対象は見落とした。小さい7個は4枚に分布し、同じ写真T0512の4個中2個で正検出。合成デモ画像3枚では4個すべてでPnP品質判定を通過。新モデルの合成285個全体、実写の測距精度、スマートフォンとWebGPUは未評価。

## 学習と再現

COCOで事前学習したtorchvision FCOSの分類出力を1クラスに交換し、TraConの既存グループ分割で微調整。実写356枚を訓練、107枚を検証に使用。RTX 5080で10エポック、バッチ4、AdamW、初期学習率0.00012、左右反転と明るさの拡張を使用。しきい値0.50で検証F1が最も高い7エポック目を選んだ（正検出201、誤検出10、見落とし69、96px未満98/156）。V10との差にはモデル、入力サイズ、事前学習と学習条件の違いを含む。

リポジトリ直下で次を実行する。第1引数はV12開発キット内の training/cone_vision_tracon_v11/datasets/tracon_v10、第2引数は上流TraConのZIP。画像ハッシュを照合する。既存の出力を保管するため、新しい出力ディレクトリを指定する。

```sh
python scripts/train_fcos.py PATH_TO_SPLITS Cone-Detection-main.zip OUTPUT_DIR
```

学習環境はPython 3.10、PyTorch 2.7.1＋CUDA 12.8、torchvision 0.22.1、OpenCV 4.7.0、ONNX 1.16.2、NumPy、Pillow。初回にtorchvisionの公式重みを取得する。best.pt、検証履歴・テスト集計、画像別の予測、FP32のcone_fcos.onnx、Web配布用cone_detector.onnxを出力する。

## 入出力と配布物

入力 images はRGB float32 [1,3,512,512]、値域0～1、余白114/255。正規化はグラフ内。出力 predictions は [1,5456,6] で、入力座標系の中心x・中心y・幅・高さ・スコア・定数1。スコアは分類確率と中心度の積の平方根。配布用モデルは大きい重みをFP16で保存し、演算前にFP32へ戻す。サイズ64,569,685バイト、SHA-256は 877a6651b8ac7082227051ab6144d13b970517997799af1c491303ebcf368b3b。

アーキテクチャとCOCO初期重みはtorchvision FCOS_ResNet50_FPN_Weights.COCO_V1。実写学習データは既存キットのTraConアノテーションと公開元の写真。写真と事前学習重みの来歴はそれぞれの配布元を参照。

- [torchvision FCOS](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html)
- [torchvisionのライセンス](https://github.com/pytorch/vision/blob/main/LICENSE)
- [TraCon](https://github.com/ikatsamenis/Cone-Detection)
