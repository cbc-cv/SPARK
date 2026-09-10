"""Evaluate the frozen SPARK XD-Violence Event-based inference models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent

# Integrity fingerprints, not training settings or learned parameters.
MODEL_SHA256 = {
    "client_0.pt": "a49776c6537a88c7a07e0007e36d1997f27ff52390d9c91cd08b38919e97b80d",
    "client_1.pt": "4f66799bd0441ad56ce2244be8d5663c97bf97ba12c32b70ef7a6a90879f5533",
    "client_2.pt": "0f167d9535eea8fee1c25287fabc1de311dc35274bde7d8c468602327bb4e8cf",
    "client_3.pt": "3bb16bb80757ec4b674d97dde631b3a5fc0c356434b4977ec9c485db39bc3089",
    "client_4.pt": "6bd0ec70a78d042ff29612c4674ec72b2aab7024e72830e599ade9e92db4c11f",
    "client_5.pt": "b29dd7ebfa6a9c940e12cf00760f437f5ba5038e338cae00b6a31ca239e20fa8",
    "text_encoder.pt": "e753de1be1ed68f6bd457ee42d473a7b13986af337b674bb5c3f11504975942c",
}
TEST_SHA256 = (
    "f97d1f76e228abbd7eda41b86ac885426cf0d28079b94f4af2511aca214bdd3c",
    "fc6798c8356bd605bc1465496767d144c196a45ff00716b87fe356c30bbea6ac",
    "2c4176f14c16a013f90abb368f9b664f5100effa87baaad7ce948c5124c88e1e",
    "ea2d17e82cc07a9996b12aca7777731c135f34dd762f2bf6c5a288658a136e97",
    "6a38ce155366edd940a9a8c2d1d1d01f82dcb5a3a9cd19bce1c70b877642c9b0",
    "2181e5cce960154a90eef5d080005ff78bc8c67c1e2584c584bb8eb4dea058a4",
)
ANNOTATION_SHA256 = "149f99e1244f6b18ef4564f18a109201d0bbb323f3c55fdd5cb5c9f04231ac43"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing release file: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path.name}. Download the complete release; for models, run git lfs pull")


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["video", "label"]:
            raise ValueError(f"Expected video,label columns: {path}")
        rows = list(reader)
    for row in rows:
        name = row["video"]
        if not name or name != Path(name).name or "/" in name or "\\" in name or not name.endswith(".npy"):
            raise ValueError(f"Invalid feature basename: {name}")
    return rows


def load_protocol(root):
    root = Path(root)
    sites = []
    for client_id, expected in enumerate(TEST_SHA256):
        path = root / f"data/splits/xd_event/test/client_{client_id}.csv"
        check_file(path, expected)
        sites.append(read_rows(path))
    names = [row["video"] for rows in sites for row in rows]
    if len(names) != 800 or len(set(names)) != len(names):
        raise ValueError("Expected 800 distinct test videos")
    path = root / "data/annotations/xd_test_frame_labels.npz"
    check_file(path, ANNOTATION_SHA256)
    with np.load(path, allow_pickle=False) as archive:
        gt_names, offsets, labels = archive["video_names"].tolist(), archive["offsets"], archive["frame_labels"]
    if gt_names != names or offsets.shape != (len(names) + 1,):
        raise ValueError("Annotation index does not match the test split")
    if offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != len(labels) or (np.diff(offsets) <= 0).any():
        raise ValueError("Invalid annotation offsets")
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Frame labels must be binary")
    return sites, {name: labels[offsets[i]:offsets[i + 1]] for i, name in enumerate(names)}


def split_features(features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 512 or features.shape[0] == 0:
        raise ValueError(f"Expected nonempty (segments, 512) features, got {features.shape}")
    if features.dtype.kind != "f" or not np.isfinite(features).all():
        raise ValueError("Features must contain finite floating-point values")
    length = len(features)
    # Preserve the released model's padding convention, including exact multiples.
    chunks = length // 256 + 1
    result = np.zeros((chunks * 256, 512), dtype=np.float32)
    result[:length] = features
    if not np.isfinite(result).all():
        raise ValueError("Features overflow float32")
    return result.reshape(chunks, 256, 512), length


def validate_features(directory, sites, ground_truth):
    directory = Path(directory)
    missing, count, segments = [], 0, 0
    for rows in sites:
        for row in rows:
            name = row["video"]
            path = directory / name
            if not path.is_file():
                missing.append(name)
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            split_features(array)
            if len(ground_truth[name]) != len(array) * 16:
                raise ValueError(f"Feature/annotation length mismatch: {name}; expected 16 frames per segment")
            count += 1
            segments += len(array)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} feature files under {directory}; first entries: {missing[:5]}")
    return {"test_videos": count, "segments": segments, "frames": segments * 16}


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be nonempty, equal-length vectors")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("AUC requires both normal and anomalous frames")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Anomaly probabilities must be finite and in [0, 1]")
    return {"auc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores))}


def summarize_client(client_id, site_arrays, scope="all"):
    metrics = {site: binary_metrics(labels, scores) for site, (labels, scores) in site_arrays.items()}
    local = metrics[client_id]
    result = {"client": client_id, "local_auc": local["auc"], "local_ap": local["ap"]}
    if scope == "all":
        if sorted(site_arrays) != list(range(6)):
            raise ValueError("Overall/Cross-site evaluation requires all six test sites")
        overall = binary_metrics(np.concatenate([site_arrays[i][0] for i in range(6)]),
                                 np.concatenate([site_arrays[i][1] for i in range(6)]))
        for name in ("auc", "ap"):
            result[f"overall_{name}"] = overall[name]
            result[f"cross_{name}"] = float(np.mean([metrics[i][name] for i in range(6) if i != client_id]))
    return result


def check_models(directory, client_ids):
    for name in ["text_encoder.pt", *[f"client_{i}.pt" for i in client_ids]]:
        check_file(Path(directory) / name, MODEL_SHA256[name])


def evaluate(args, sites, gt):
    import torch

    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("Use cpu or cuda:0; select another physical GPU with CUDA_VISIBLE_DEVICES")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the matching PyTorch build or specify --device cpu")
    torch.set_num_threads(2)
    model_directory = ROOT / "models"
    # File objects also support non-ASCII paths in Windows PyTorch builds.
    with (model_directory / "text_encoder.pt").open("rb") as handle:
        encoder = torch.jit.load(handle, map_location=args.device).eval()
    rows = []
    with torch.inference_mode():
        for client_id in args.client:
            with (model_directory / f"client_{client_id}.pt").open("rb") as handle:
                client = torch.jit.load(handle, map_location=args.device).eval()
            arrays = {}
            started = time.monotonic()
            targets = range(6) if args.scope == "all" else [client_id]
            for site_id in targets:
                labels, predictions = [], []
                for row in sites[site_id]:
                    name = row["video"]
                    feature = np.load(Path(args.features) / name, allow_pickle=False)
                    chunks, length = split_features(feature)
                    # All chunks of one video must be evaluated together.
                    visual = torch.from_numpy(chunks).to(args.device)
                    visual_state, prompts = client.encode(visual)
                    logits = client.score(visual_state, encoder(prompts))
                    if tuple(logits.shape) != (len(chunks), 256, 7):
                        raise RuntimeError(f"Unexpected logits shape for {name}: {tuple(logits.shape)}")
                    scores = 1 - logits.reshape(-1, logits.shape[-1])[:length].softmax(dim=-1)[:, 0]
                    scores = np.repeat(scores.cpu().numpy(), 16)
                    if len(scores) != len(gt[name]):
                        raise ValueError(f"Frame count mismatch: {name}")
                    labels.append(gt[name])
                    predictions.append(scores)
                arrays[site_id] = (np.concatenate(labels), np.concatenate(predictions))
                print(f"Client {client_id}: evaluated site {site_id} ({len(sites[site_id])} videos)", flush=True)
            row = summarize_client(client_id, arrays, args.scope)
            rows.append(row)
            print(f"Client {client_id}: Local AUC={row['local_auc'] * 100:.4f}% AP={row['local_ap'] * 100:.4f}%"
                  f" ({time.monotonic() - started:.1f}s)", flush=True)
            del client
    names = [key for key in rows[0] if key != "client"]
    return {"dataset": "XD-Violence", "partition": "Event-based",
            "topology": "Fully Connected", "scope": args.scope,
            "evaluated_clients": args.client, "metric_units": "fraction",
            "clients": rows, "mean": {name: float(np.mean([row[name] for row in rows])) for name in names}}


def output_path(path):
    destination = Path(path).resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("Evaluation output must be a .json file")
    if destination.is_relative_to(ROOT) and not destination.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Write results to outputs/ or outside the release; do not overwrite release files")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="XDTestClipFeatures directory containing the 800 test .npy files")
    parser.add_argument("--client", type=int, nargs="+", choices=range(6), default=list(range(6)))
    parser.add_argument("--scope", choices=["all", "local"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-data", action="store_true", help="Check all test features and annotations without running a model")
    checks.add_argument("--check-models", action="store_true", help="Check model SHA256 without importing PyTorch")
    args = parser.parse_args()
    if len(set(args.client)) != len(args.client):
        parser.error("--client entries must be unique")
    if args.check_models:
        check_models(ROOT / "models", args.client)
        print("Model checksums: OK")
        return
    if args.features is None:
        parser.error("--features is required")
    try:
        destination = output_path(args.output)
    except ValueError as error:
        parser.error(str(error))
    sites, gt = load_protocol(ROOT)
    status = validate_features(args.features, sites, gt)
    print("Data check: " + json.dumps(status), flush=True)
    if args.check_data:
        return
    check_models(ROOT / "models", args.client)
    report = evaluate(args, sites, gt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMean across evaluated clients (percent):")
    for name, value in report["mean"].items():
        print(f"  {name}: {100 * value:.4f}")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
