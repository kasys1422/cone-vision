"""Stage the verified model release ZIPs for local serving or GitHub Pages."""

import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


TAG = "depth-da3-q4-v1"
BASE_URL = f"https://github.com/kasys1422/cone-vision/releases/download/{TAG}"
ASSETS = {
    "da3-base-q4.zip": "5ec0c5fe61f49b9a12cf3e1250491b02edc2fdb6cbcf5f35a8b238c01d4d431d",
    "da3-large-1.1-q4.zip": "4f2c8063841f497442dc44221831cb124307581636fed8a251f039fc04d3e88d",
}


def stage(root):
    directory = root / "models" / "depth" / TAG
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        digest = hashlib.sha256()
        with NamedTemporaryFile(dir=directory, delete=False) as tmp:
            staging = Path(tmp.name)
        try:
            with urlopen(f"{BASE_URL}/{name}", timeout=120) as response, staging.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Release ZIP SHA-256 mismatch: {name}")
            staging.replace(directory / name)
            print(f"Verified {name}", flush=True)
        finally:
            staging.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    stage(parser.parse_args().site_root)
