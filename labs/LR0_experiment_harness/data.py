import hashlib
import json
import yaml
import argparse
from pathlib import Path

import sklearn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
data_dir = root / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = root / config_path

with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

split_seed = config["split_seed"]

_, y = load_digits(return_X_y=True)
indices = np.arange(len(y))

rest, test = train_test_split(
    indices,
    test_size=0.2,
    stratify=y,
    random_state=split_seed
)
train, val = train_test_split(
    rest,
    test_size=0.25,
    stratify=y[rest],
    random_state=split_seed
)

split = {
    "split_seed": split_seed,
    "train": train.tolist(),
    "val": val.tolist(),
    "test": test.tolist()
}

split_path = data_dir / "split.json"
split_path.write_text(
    json.dumps(split, indent=2) + "\n",
    encoding="utf-8"
)
sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()

manifest = {
    "source": "scikit-learn load_digits",
    "scikit_learn_version": sklearn.__version__,
    "split_rule": (
        "Stratified 80/20 train+val/test split; "
        "then stratified 75/25 train/val split; random_state=0"
    ),
    "split_seed": split_seed,
    "sizes": {
        "train": len(train),
        "val": len(val),
        "test": len(test)
    },
    "split_sha256": sha256
}

(data_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

assert len(train) == 1077
assert len(val) == 360
assert len(test) == 360
assert len(set(train) & set(val)) == 0
assert len(set(train) & set(test)) == 0
assert len(set(val) & set(test)) == 0
assert set(train) | set(val) | set(test) == set(indices)

for name, part in [("train", train), ("val", val), ("test", test)]:
    assert set(np.unique(y[part])) == set(range(10)), name

print(f"Saved {split_path}")
print(f"Sizes: train={len(train)}, val={len(val)}, test={len(test)}")
print(f"SHA-256: {sha256}")