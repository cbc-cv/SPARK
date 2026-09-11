# Model Weights

Model weights are not stored in this Git repository.

Please download the weight archive from the GitHub Releases page and extract it
into this directory. After extraction, this directory should contain:

```text
client_0.pt
client_1.pt
client_2.pt
client_3.pt
client_4.pt
client_5.pt
text_encoder.pt
```

Then run evaluation from the repository root:

```bash
python evaluate.py --features /path/to/XDTestClipFeatures
```
