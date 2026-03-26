import os
import csv
import yaml
import json
import numpy as np
import itertools
import tensorflow_datasets as tfds



def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def write_pairs_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["left_path", "right_path", "label", "split"]
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def reconstruct_identity_split(config, seed):
    
    ds = tfds.load("lfw", split="train", data_dir=config["data_dir"])

    identities = set()
    for ex in tfds.as_numpy(ds):
        identities.add(ex["label"].decode("utf-8"))

    identities = sorted(identities)

    rng = np.random.default_rng(seed)
    rng.shuffle(identities)

    tr = float(config.get("train_frac", 0.70))
    va = float(config.get("val_frac", 0.15))
    te = float(config.get("test_frac", 0.15))

    n = len(identities)
    n_tr = int(tr * n)
    n_va = int(va * n)

    tr_ids = set(identities[:n_tr])
    va_ids = set(identities[n_tr:n_tr + n_va])
    te_ids = set(identities[n_tr + n_va:])

    return tr_ids, va_ids, te_ids

def collect_images_by_identity(cfg, split_ids):
    
    ds = tfds.load("lfw", split="train", data_dir=cfg["data_dir"])

    images_by_id = {}

    per_id_index = {}

    for ex in tfds.as_numpy(ds):
        identity = ex["label"].decode("utf-8")

        if identity not in split_ids:
            continue

        per_id_index[identity] = per_id_index.get(identity, 0) + 1

        filename = f"{identity}_{per_id_index[identity]:04d}.jpg"

        # Save paths relative to dataset cache dir
        rel_path = os.path.join("lfw", identity, filename)

        images_by_id.setdefault(identity, []).append(rel_path)

    # Sort so it is deterministic
    for identity in images_by_id:
        images_by_id[identity] = sorted(images_by_id[identity])

    return images_by_id


def generate_pairs(images_by_id, seed, split_name):
    rng = np.random.default_rng(seed)

    identities = sorted(images_by_id.keys())

    positive_pairs = []
    negative_pairs = []

    # Positive pairs
    for identity in identities:
        imgs = images_by_id[identity]

        #Improvement- cap images per identity
        if len(imgs) > 10:
            imgs = imgs[:10]
        if len(imgs) < 2:
            continue

        combos = list(itertools.combinations(imgs, 2))
        combos = sorted(combos)

        positive_pairs.extend(combos)

    # Negative pairs
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id_a = identities[i]
            id_b = identities[j]

            img_a = images_by_id[id_a][0]
            img_b = images_by_id[id_b][0]

            negative_pairs.append((img_a, img_b))

    negative_pairs = sorted(negative_pairs)

    # Balance negatives and positives
    if len(negative_pairs) > len(positive_pairs):
        rng.shuffle(negative_pairs)
        negative_pairs = negative_pairs[:len(positive_pairs)]

    rows = []

    for left, right in positive_pairs:
        rows.append({
            "left_path": left,
            "right_path": right,
            "label": 1,
            "split": split_name
        })

    for left, right in negative_pairs:
        rows.append({
            "left_path": left,
            "right_path": right,
            "label": 0,
            "split": split_name
        })

    # Final deterministic ordering
    rows = sorted(rows, key=lambda x: (x["label"], x["left_path"], x["right_path"]))

    return rows



def main(config_path: str):
    config = load_config(config_path)
    manifest = load_manifest(os.path.join(config["outputs_dir"], "manifest.json"))

    seed = manifest["seed"]

    tr_ids, va_ids, te_ids = reconstruct_identity_split(config, seed)

    split_map = {
        "train": tr_ids,
        "val": va_ids,
        "test": te_ids
    }

    
    for split_name, split_ids in split_map.items():

        images_by_id = collect_images_by_identity(config, split_ids)

        rows = generate_pairs(images_by_id, seed, split_name)

        outpath = os.path.join(
            config["outputs_dir"],
            "pairs",
            f"{split_name}.csv"
        )

        write_pairs_csv(outpath, rows)

        print(f"Wrote {len(rows)} pairs -> {outpath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)
