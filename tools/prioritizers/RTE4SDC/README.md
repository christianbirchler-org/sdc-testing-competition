# RTE4SDC

Ranking Transformer Encoder for SDC test case prioritization.

The pre-trained ONNX model (`rte4sdc.onnx`) is used by `main.py` to serve
prioritisation requests via gRPC.

---

## Training pipeline

### 1) Setting the environment

```bash
cd train
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2) Preparing the dataset

Training requires the [SensoDat dataset](https://github.com/christianbirchler-org/sensodat/).

```bash
mkdir -p data
curl -L -o data/data.zip \
  "https://media.githubusercontent.com/media/christianbirchler-org/sensodat/main/data/data.zip"
unzip data/data.zip -d data/
python extract_sensodat.py --sensodat-dir data --output data/sensodat.json
```

### 3) Train

```bash
python train.py --config config.yaml
```

### 4) Export ONNX

```bash
python export.py \
  --config config.yaml \
  --checkpoint output/best.pt \
  --output ../rte4sdc.onnx
```