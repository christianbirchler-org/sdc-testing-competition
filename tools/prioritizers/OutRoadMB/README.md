# OutRoadMB

**Outlier-based Road Geometry Prioritizer for Self-Driving Car Test Suites**

## Approach

OutRoadMB prioritizes SDC test cases by detecting **geometric outliers** in road shapes. The intuition: roads with unusual geometry (sharp turns, abnormal curvature, low safety margins) are more likely to trigger failures in the self-driving car simulator.

### Feature Extraction

For each test case, we extract 8 road geometry features:

| # | Feature | Description |
|---|---------|-------------|
| F1 | Direct Distance | Euclidean distance between road start and end |
| F2 | Road Distance | Total road segment length |
| F6 | Total Angle | Sum of all angle changes along the road |
| F9 | Max Angle | Maximum single angle change |
| F11 | Mean Angle | Average angle change |
| F8 | Std Angle | Standard deviation of angle changes |
| F14 | Road Safety Sum | Total area between road curve and simplified polyline (inflection-point detection + Shoelace formula) |
| F15 | Road Safety Mean | Mean safety area per road segment |

### Outlier Detection

Test cases are ranked by their distance from the feature-space centroid using one of two metrics:

- **Mahalanobis distance** (default): accounts for feature correlations via inverse covariance matrix. More robust when features are correlated.
- **Euclidean distance**: z-score normalized. Simpler, treats features as independent.

Tests with the highest anomaly scores (most unusual road geometry) are prioritized first.

### Why Unsupervised?

The approach requires no training labels — it works entirely from road geometry. This makes it robust to distribution shifts across different test campaigns and generators (Frenetic, AmbiGen, etc.).

## Available Strategies

| Key | Description |
|-----|-------------|
| `mahalanobis-outlier-first` | **(default)** Mahalanobis distance outlier detection |
| `euclidean-outlier-first` | Euclidean z-score outlier detection |
| `less-safe-first` | Rank by road safety area (Shoelace deviation) |
| `longest-first` | Baseline: most road points first |
| `total-distance-first` | Baseline: longest road distance first |

## Docker Image

> **Pre-built image:** `marcellobabbi/outroadmb:latest`

### Build locally

```bash
docker build -t outroadmb .
```

### Run

```bash
# Default strategy (mahalanobis-outlier-first)
docker run --rm outroadmb -p 50051

# Specific strategy
docker run --rm outroadmb -p 50051 -s euclidean-outlier-first

# Via environment variable
docker run --rm -e STRATEGY=less-safe-first -e GRPC_PORT=50051 outroadmb
```

## Project Structure

```
OutRoadMB/
├── main.py                     # gRPC server (CompetitionTool interface)
├── strategies.py               # Domain logic: feature extraction + prioritization strategies
├── competition_2026_pb2.py     # Generated protobuf stubs
├── competition_2026_pb2_grpc.py# Generated gRPC stubs
├── Dockerfile
├── LICENSE.md
└── README.md
```

## License

MIT — see [LICENSE.md](LICENSE.md)
