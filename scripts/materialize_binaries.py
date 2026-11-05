"""Restore large artifacts from verified base64-gzip chunks. No network required."""
import base64
import gzip
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ("detector", "models/retrain_v7/cone_detector.onnx", "305fa8cf092a1d8cc4896cf8a0c8ccaa14f412a8368a945a50d73318332d2430"),
    ("opencv", "web_demo/vendor/opencv.js", "63366510248adf3a7eddf3e793dd825404efb7df3749f4d6f8557c7fa4ca8aa0"),
)


def main():
    for name, destination, expected_sha256 in ARTIFACTS:
        fragments = sorted((ROOT / ".payload" / name).glob("part-*.txt"))
        if len(fragments) != 5:
            raise RuntimeError(f"{name}: expected 5 fragments, found {len(fragments)}")
        encoded = "".join(part.read_text(encoding="ascii").strip() for part in fragments)
        original = gzip.decompress(base64.b64decode(encoded, validate=True))
        sha256 = hashlib.sha256(original).hexdigest()
        if sha256 != expected_sha256:
            raise ValueError(f"{name}: SHA-256 mismatch {sha256}")
        output = ROOT / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(original)
        print(f"Restored {destination}: {len(original)} bytes, SHA-256 {sha256}")


if __name__ == "__main__":
    main()
