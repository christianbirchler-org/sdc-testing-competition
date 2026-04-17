# RoadFury — SDC Test Prioritizer

**RoadFury** is a test prioritization tool for simulation-based regression testing of self-driving cars (SDCs). Developed for the ICST/SBFT 2026 SDC Testing Competition.

## Approach

RoadFury uses a **Transformer Encoder with Stochastic Weight Averaging (SWA)** to predict which simulation tests are most likely to result in FAIL (car driving off-lane), prioritizing those first to maximize early fault detection.

### Architecture

1. **10-Channel Road Geometry Features** (per road point):
   - Segment length, absolute angle change, Menger curvature
   - Curvature jerk (1st derivative), curvature acceleration (2nd derivative)
   - Cumulative distance (normalized), heading sin/cos
   - Relative position, local curvature std (window=11)

2. **Transformer Encoder**:
   - Linear input projection (10 → 128) with LayerNorm + GELU
   - Learnable [CLS] token + positional embeddings
   - 4-layer Pre-LN TransformerEncoder (d=128, 8 heads, FFN=512)
   - CLS-only pooling → MLP classifier (128 → 64 → 1)

3. **Stochastic Weight Averaging (SWA)**:
   - Trains for 75 epochs; averages weights from epoch 50–75
   - Finds flatter minima → better generalization on unseen roads
   - Izmailov et al. (2018): "Averaging Weights Leads to Wider Optima"

4. **Training Data**: 36,006 labeled tests from SensoDat (28,804 train / 7,202 val)

### Performance

| Metric | RoadFury (Transformer+SWA) | ITEP4SDC (SOTA) | Random |
|--------|---------------------------|-----------------|--------|
| APFD (multi-trial) | **0.8042 ± 0.0120** | 0.7812 | 0.4930 |

## Usage

### Docker (recommended)

```bash
# Build
docker build -t road-fury .

# Run on port 4545
docker run --rm -t -p 4545:4545 road-fury -p 4545
```

### Local

```bash
pip install -r requirements.txt
python main.py -p 4545
```

## Files

| File | Description |
|------|-------------|
| `main.py` | gRPC server — implements `Name`, `Initialize`, `Prioritize` |
| `features.py` | 10-channel road geometry feature extraction |
| `roadfury_best.pt` | Pre-trained Transformer+SWA model weights |
| `Dockerfile` | Docker container configuration |
| `idea-exps/` | Training & experiment scripts (run on Kaggle H100) |

## References

[1] C. Birchler et al., "Cost-effective Simulation-based Test Selection in Self-driving Cars Software with SDC-Scissor," SANER 2022.

[2] C. Birchler et al., "Machine learning-based test selection for simulation-based testing of self-driving cars software," EMSE 28, 71 (2023).

[3] P. Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," UAI 2018.
