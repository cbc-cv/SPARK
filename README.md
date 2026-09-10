# SPARK: XD-Violence Event-based Evaluation

[中文说明](#中文说明) 

## 中文说明

本目录仅提供新训练的 **XD-Violence / Event-based / Fully Connected** 模型测试。
只有一个 Python 文件 `evaluate.py`

### 文件

```text
SPARK_main/
  evaluate.py                 # 数据读取、模型推理、AUC/AP 计算
  models/
    client_0.pt ... client_5.pt
    text_encoder.pt
  data/
    splits/xd_event/train/     # 现成的训练划分 CSV
    splits/xd_event/test/      # 现成的测试划分 CSV
    annotations/              # 计算 AUC/AP 必需的帧级标注
  requirements.txt
  third_party/                # 第三方声明
```

CSV 仅包含特征文件名和标签。客户端 0 至 5 的事件依次为 Fighting、Riot、Abuse、Shooting、Explosion、Car accident。
测试只读取测试 CSV，不读取训练 CSV，也不需要训练特征。
`.pt` 已包含权重和推理计算图，不需要原始模型 Python 实现或另外下载完整 CLIP 权重。

### 运行

需要与实验一致的 CLIP ViT-B/16 测试特征：800 个 `.npy` 文件直接放在同一目录，
每个文件形状为 `(片段数, 512)`，每个片段对应 16 帧。不提供原始视频或特征文件。
特征来源参考 [VadCLIP Setup](https://github.com/nwpu-zxr/VadCLIP#setup)；请使用匹配的特征，不能仅凭维度判断一致。

参考环境为 Python 3.12、PyTorch 2.5.1+cu121。进入本目录后执行：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python evaluate.py --features /path/to/XDTestClipFeatures
```

默认用 `cuda:0` 测试六个模型；无 GPU 时加 `--device cpu`。
Linux 上可用 `CUDA_VISIBLE_DEVICES=0` 选择物理 GPU。
仅测试单个模型的本地数据时，加 `--client 0 --scope local`。
文件检查可用 `--check-models`，或 `--features /path/to/XDTestClipFeatures --check-data`。

结果默认写入 `outputs/evaluation.json`；终端显示百分数，JSON 内为 0 到 1。
`--features`、`--device`、`--client`、`--scope`、`--output` 仅控制测试，不是训练参数。
模型和测试文件的 SHA256 校验已内置在脚本中，不再需要额外配置文件。

### 指标

| 统计方式 | AUC (%) | AP (%) |
| --- | ---: | ---: |
| Local | 90.7924 | 64.2509 |
| Overall | 93.3098 | 76.1962 |
| Cross-site | 90.6923 | 63.6352 |

- Local：各模型在自己的测试集上计算指标，再对六个模型求平均。
- Overall：各模型在全部测试集的并集上计算指标，再对六个模型求平均。
- Cross-site：各模型在另外五个测试集上分别计算指标并取平均，再对六个模型求平均。

AP 使用 `average_precision_score`，不是 PR 曲线梯形积分。
片段分数为 `1 - softmax(logits)[normal]`，重复 16 次后与帧级标注比较。
保持现有的 256 片段分块及补零规则，同一视频的所有块一起推理。
以上为这批权重的复测结果；脚本通过实际推理计算指标，不读取预存结果。
