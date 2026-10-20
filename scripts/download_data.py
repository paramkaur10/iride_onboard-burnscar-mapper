"""Data and trained models of the paper, from the private Hugging Face dataset, into data/ and models/.

    hf auth login                                   # an account with access, or HF_TOKEN=<token> in the environment
    python scripts/download_data.py                 # everything, about 68 GB
    python scripts/download_data.py --only models   # parts: heo phisat2 s2 effis tables pools models notebooks
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "neet1797/iride-heo-burnscar"
PARTS = {
    "heo": ["data/heo/*"],
    "phisat2": ["data/phisat2_sim/*"],
    "s2": ["data/s2_test/*"],
    "effis": ["data/effis/*"],
    "tables": ["data/tables/*", "data/README.md"],
    "pools": ["data/pool_v2/*"],
    "models": ["models/*"],
    "notebooks": ["notebooks_executed/*"],
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(PARTS), help="download only these parts")
    a = ap.parse_args()
    parts = a.only or list(PARTS)
    snapshot_download(REPO, repo_type="dataset", local_dir=ROOT, allow_patterns=[p for k in parts for p in PARTS[k]])
    for z in sorted((ROOT / "data" / "heo" / "products_zip").glob("*.zip")):     # unzipped, as notebook 03 reads them
        out = ROOT / "data" / "heo" / "products" / z.stem
        if not out.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(out)
            print("unzipped", z.name)
    print("downloaded:", ", ".join(parts))


if __name__ == "__main__":
    main()
