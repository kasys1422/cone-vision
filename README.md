# Cone Vision

ブラウザ内で交通コーンを検出し、先端＋台座4隅の5点から距離・姿勢を推定する実験的なWebアプリです。GitHub Pagesでの静的配信を想定しています。

## 現在の状態

- **リポジトリ初期化段階**。Webデモ・ONNXモデルは、まだこのリポジトリに含まれていません。
- 採用候補は **v7**（物体検出：YOLOX-Nano、特徴点：独立した軽量CNN）。学習済みモデルは別途作成・評価済みです。
- **実写向け完成版ではありません。** v7は合成テストで改善しましたが、実写画像では背景への誤検出が残り、以前のモデルより誤検出が増えた例があります。
- 実寸とカメラ校正がない画像には、メートル単位の距離を表示しない方針です。

## 計画する機能

- 写真／カメラ映像からの複数コーン検出
- BBox・5点キーポイントの重ね合わせ
- カメラ内部パラメータと実物寸法に基づくPnP
- 信頼性が低い測距の保留、および結果JSONの出力
- ONNX Runtime Web（WASM・対応ブラウザのWebGPU）とOpenCV.js
- サーバーで画像を処理しないクライアント側推論

## 配置予定

```text
cone-vision/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── assets/
│   └── cone_geometry.json
├── models/
│   └── retrain_v7/
│       ├── cone_detector.onnx
│       ├── cone_detector.interface.json
│       ├── cone_keypoints.onnx
│       └── cone_keypoints.interface.json
├── web_demo/
│   ├── index.html
│   ├── inference.mjs
│   ├── pnp.mjs
│   ├── vendor/
│   └── samples/
└── docs/
    ├── DEPLOYMENT.md
    └── MODEL_STATUS.md
```

実際の配置と公開前の確認方法は [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) を参照してください。**READMEに構成が記載されていても、アップロードされていないモデル／Webアプリが存在することを意味しません。**

## ライセンスと素材

独自ソースコード: GPL-3.0-only（[LICENSE](LICENSE)）。学習済みモデル・提供3D素材・第三者ライブラリには別の利用条件があるため、モデルと素材を追加する際に出典・ライセンス・変更履歴も同時に収録してください。

## 注意

研究・試作向けです。安全用途・規制用途・車両運転支援等に使用しないでください。正確な実写測距には実寸の測定、カメラ校正、独立した実写評価が必要です。
