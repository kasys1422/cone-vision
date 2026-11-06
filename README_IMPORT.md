# Webアプリの起動

ZIPを展開し、`index.html` があるディレクトリで `python -m http.server 8877` を実行して `http://localhost:8877/` を開いてください。GitHub Pagesでは展開内容をリポジトリのルートに置きます。OpenCVの分割データを格納する `.payload/` と `.nojekyll` も必要です。

カメラを利用する場合はHTTPSまたはlocalhostを使います。初回はONNX Runtime WebをCDNから読み込みます。
