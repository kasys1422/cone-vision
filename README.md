# Cone Vision

スマートフォンのカメラや写真から交通コーンを検出し、選択したコーンの5点特徴点・距離を表示する **実験用PWA** です。推論は端末内で動作します。

## 写真のEXIFから画角を自動設定

写真を読み込むと、35mm判換算焦点距離などのEXIFから水平画角を推定し、PnPのカメラ内部パラメータへ概算値を反映します。スキャン前に値を確認・手動変更できます。カメラ校正JSONがある場合は校正値を優先します。詳しくは [EXIF画角推定の説明](docs/EXIF_FOV.md)。

## 操作

1. 「カメラで撮影する」または「ギャラリーから画像を選択」
2. 写真を確認して「スキャンする」
3. 検出されたコーンを画像上または一覧から選択
4. 距離、奥行き、検出スコアを確認

GitHub Pagesにデプロイすると `https://kasys1422.github.io/cone-vision/` で利用できます。

## ローカル実行

```bash
python -m http.server 8877
```

`http://localhost:8877/` を開きます。カメラは原則HTTPSまたはlocalhostが必要です。初回はONNX Runtime WebをCDNから読み込むためインターネット接続が必要です。

## モデル

- **物体検出器**: YOLOX-Nano、RGB 256×256入力 (`models/retrain_v7/cone_detector.onnx`)
- **5点推定器**: 軽量CNN、RGB 96×96 ROI入力 (`models/retrain_v7/cone_keypoints.onnx`)
- **姿勢・距離**: OpenCV.jsによるC4対称性を考慮したPnP。計算にはコーンの実寸とカメラ内部パラメータが必要です。

`web_demo/` は静的アプリで、画像・映像をサーバーに送信しません。モデル実行はWebGPUまたはWASMです（端末・ブラウザに依存）。

## 測距に関する注意

**V7は実写向けに検証が完了したモデルではありません。** 合成データでは改善しましたが、実写の背景誤検出が残っています。初期画面の画角（HFOV）入力で求める値は概算であり、校正値ではありません。実際の距離はカメラ校正およびコーン高さ・台座幅の実測値を用いてください。

PnPの再投影誤差が小さくても、誤検出した背景に距離が出る可能性があります。研究・試作用であり、交通安全上の判断には使用できません。

## 構成

- `web_demo/`: PWA UI、カメラ、画像入力、BBox/特徴点表示、PnP、Service Worker
- `models/retrain_v7/`: V7モデル、入出力仕様、モデルカード
- `assets/cone_geometry.json`: 3D特徴点と寸法設定
- `THIRD_PARTY_NOTICES.md`: 依存関係・素材の利用条件

## ライセンス

独自コード GPL-3.0-only。モデル・3D素材等は別ライセンスです。詳細は `THIRD_PARTY_NOTICES.md` と `models/retrain_v7/MODEL_CARD.md` を参照してください。
