# SPARK: XD-Violence Event-based Evaluation

This repository provides the public evaluation package for SPARK on the
XD-Violence event-based split. It contains the inference script, data split
files, frame-level annotations required for AUC/AP calculation, and third-party
notices. Model weights are distributed through GitHub Releases.

## 中文说明

本仓库用于复现 SPARK 在 **XD-Violence / Event-based / Fully Connected** 设置下的测试结果。仓库中只保留：

- `evaluate.py`：推理与 AUC/AP 计算脚本。
- `data/splits/xd_event/`：固定的数据划分 CSV。
- `data/annotations/`：计算 AUC/AP 所需的测试集帧级标注。
- `third_party/`：第三方声明。

模型权重不直接放在 Git 仓库中。请在仓库右侧的 **Releases** 页面下载权重压缩包，并解压到：

```text
SPARK_main/models/
```

解压后目录应包含：

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

客户端 0 到 5 的事件类别依次为 Fighting、Riot、Abuse、Shooting、Explosion、Car accident。测试只读取测试 CSV，不读取训练 CSV，也不包含训练代码或 P2P 通信流程。

## Environment

Reference environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU evaluation is also supported, but slower.

## Data Preparation

You need CLIP ViT-B/16 features for the XD-Violence test videos. Put all test
feature files in one directory. Each file should be a `.npy` array with shape:

```text
(number_of_segments, 512)
```

Each segment corresponds to 16 video frames. Raw videos and feature extraction
code are not included in this release.

## Run Evaluation

After downloading and extracting the model weights from Releases:

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

The terminal prints percentages, while the JSON file stores metrics in the
range `[0, 1]`.

## Expected Results

| Evaluation scope | AUC (%) | AP (%) |
| --- | ---: | ---: |
| Local | 90.7924 | 64.2509 |
| Overall | 93.3098 | 76.1962 |
| Cross-site | 90.6923 | 63.6352 |

Metric definitions:

- Local: each client is evaluated on its own test split, then the six metrics are averaged.
- Overall: each client is evaluated on the union of all test splits, then the six metrics are averaged.
- Cross-site: each client is evaluated on the other five test splits, then averaged across clients.

AP is computed with `average_precision_score`. Segment anomaly scores are
computed as `1 - softmax(logits)[normal]`, repeated 16 times, and compared with
frame-level labels. The script computes metrics from model inference and does
not read any precomputed result file.
