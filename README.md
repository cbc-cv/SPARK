# SPARK: XD-Violence Event-based Evaluation

This repository provides the public evaluation package for SPARK on the
XD-Violence event-based split. It contains the inference script, fixed data
splits, frame-level annotations required for AUC/AP calculation, and third-party
notices. Model weights are provided separately through GitHub Releases.

## Repository Contents

```text
SPARK_main/
  evaluate.py                  # Inference and AUC/AP evaluation
  models/
    README.md                  # Weight download instructions
  data/
    splits/xd_event/train/     # Fixed training split CSV files
    splits/xd_event/test/      # Fixed test split CSV files
    annotations/               # Frame-level labels for metric calculation
  requirements.txt
  third_party/                 # Third-party notices
```

The repository does not include training code, split-generation code, or the
P2P communication procedure. The test script only reads the test split CSV files
and the provided annotations.

## Model Weights

Model weights are not stored directly in this Git repository. Please download
the weight archive from the **Releases** page and extract it into:

```text
SPARK_main/models/
```

After extraction, the directory should contain:

```text
models/
  client_0.pt
  client_1.pt
  client_2.pt
  client_3.pt
  client_4.pt
  client_5.pt
  text_encoder.pt
```

The client-event order is:

| Client | Event category |
| ---: | --- |
| 0 | Fighting |
| 1 | Riot |
| 2 | Abuse |
| 3 | Shooting |
| 4 | Explosion |
| 5 | Car accident |

The `.pt` files contain the model weights and inference graphs. No separate
training implementation or full CLIP checkpoint is required for evaluation.

## Environment

The reference environment is Python 3.12 with PyTorch 2.5.1+cu121.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU evaluation is also supported, but it is slower.

## Data Preparation

The evaluator expects CLIP ViT-B/16 features for the XD-Violence test videos.
Place all test feature files in a single directory. Each file should be a `.npy`
array with shape:

```text
(number_of_segments, 512)
```

Each segment corresponds to 16 video frames. Raw videos and feature extraction
code are not included in this release. Please use feature files that match the
evaluation protocol; matching only the feature dimension is not sufficient.

## Run Evaluation

After downloading and extracting the model weights from Releases, run:

```bash
python evaluate.py --features /path/to/XDTestClipFeatures
```

Useful options:

```bash
python evaluate.py --features /path/to/XDTestClipFeatures --device cpu
python evaluate.py --features /path/to/XDTestClipFeatures --client 0 --scope local
python evaluate.py --check-models
python evaluate.py --features /path/to/XDTestClipFeatures --check-data
```

The default output file is:

```text
outputs/evaluation.json
```

The terminal prints metrics as percentages, while the JSON file stores metrics
in the range `[0, 1]`.

## Expected Results

| Evaluation scope | AUC (%) | AP (%) |
| --- | ---: | ---: |
| Local | 90.7924 | 64.2509 |
| Overall | 93.3098 | 76.1962 |
| Cross-site | 90.6923 | 63.6352 |

Metric definitions:

- Local: each client is evaluated on its own test split, and the six metrics are averaged.
- Overall: each client is evaluated on the union of all test splits, and the six metrics are averaged.
- Cross-site: each client is evaluated on the other five test splits, and the metrics are averaged across clients.

AP is computed with `average_precision_score`. Segment-level anomaly scores are
computed as `1 - softmax(logits)[normal]`, repeated 16 times, and compared with
frame-level labels. The script computes all metrics from model inference and
does not read any precomputed result file.
