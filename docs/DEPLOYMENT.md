# WebデモをGitHub Pagesに配置する手順

リポジトリを作成しただけではWebデモは公開されません。v7の過去成果物から必要なコード・ONNXと第三者ライブラリを取り出し、GitHubへ追加した後にPagesを設定してください。

## 必要なファイル

既存の `cone_vision_retrained_v7.zip`（ルート `cone-vision/`）から、以下を**ディレクトリ構造を維持して**配置します。

- `web_demo/index.html`, `inference.mjs`, `pnp.mjs`, `samples/`
- `web_demo/vendor/opencv.js`, `OpenCV-LICENSE.txt`, `provenance.json`
- `assets/cone_geometry.json`
- `models/retrain_v7/cone_detector.onnx`, `cone_detector.interface.json`
- `models/retrain_v7/cone_keypoints.onnx`, `cone_keypoints.interface.json`
- `THIRD_PARTY_NOTICES.md` と必要なライセンス通知

**重要**：元デモはv3など複数モデルの切替を前提にしたUIです。v7だけを公開する場合は、`web_demo/index.html` のv3・旧・Coneslayer選択肢を削除し、v7を選択状態に変更してください。古い未配置モデルの選択肢を残すと404になります。標準のROI／PnP処理が依存するOpenCV.jsも省略しないでください。

学習用画像、Optimizerの重み、巨大なデータZIPはPages配信に不要です。モデル重みを公開する場合はモデルカードと出典／ライセンスを必ず同梱してください。

## ローカル確認

```bash
python -m http.server 8877
```

`http://localhost:8877/web_demo/` を開きます。まず赤・黄・複数のサンプル画像を実行し、AとBが読み込まれ、BBoxと5点が表示されるか確認してください。PnPの距離は付属する合成サンプルの既知校正値で動作確認し、実写での正確さを保証するものではありません。

## 公開

GitHubリポジトリの **Settings → Pages → Build and deployment → Deploy from a branch → main / (root)** を設定します。公開後の想定パスは `https://kasys1422.github.io/cone-vision/web_demo/` です（Pagesが有効化されるまではアクセスできません）。

ONNX Runtime Webは当初 `onnxruntime-web@1.22.0` をCDNから取得する構成で、利用するOpenCV.jsは別配布のバイナリを含みます。依存ライブラリの更新時は数値一致テストが必要です。

## 未検証

v7のスマートフォン実機、実ブラウザWebGPU、校正済み実写距離精度は未確認。公開前にメニューの404、WebGPU失敗時のWASM切替、カメラ権限、スマートフォンUIを確認してください。
