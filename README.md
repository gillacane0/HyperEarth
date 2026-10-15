**HyperEarth**

HyperEarth is a framework with the objective of testing and comparing different deep-learning model on differents datasets dor the Earth Observation, specifically for Hyper Spectral type of data.

Intended structure for now:

```text
hyperspectral-benchmarking/
├── src/                            # Source code
│   ├── data/                       # Data handling
│   │   ├── loaders.py              # Loading dataset script
│   │   ├── preprocessing.py        # Preprocessing data
│   │   └── utils.py
│   ├── models/                     # Models definitions
│   │   ├── baselines/              # Classical ML models
│   │   │   ├── svm.py
│   │   │   ├── random_forest.py
│   │   │   └── knn.py
│   │   ├── cnn/                    # Convolutional networks
│   │   │   ├── cnn_1d.py
│   │   │   ├── cnn_2d.py
│   │   │   └── cnn_3d.py
│   │   ├── transformers/           # Transformer-based models
│   │   │   ├── vit.py
│   │   │   └── spectralformer.py
│   │   ├── hybrids/                # Hybrid architectures
│   │   │   ├── cnn_lstm.py
│   │   │   └── cnn_transformer.py
│   │   └── utils.py
│   ├── experiments/                # Training and evaluation
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── benchmark.py
│   └── utils/                      # Shared helpers
│       ├── logging.py
│       └── visualization.py
│
├── data/                           # Dataset and inputs data
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── results/                        # Experiment outputs
│   ├── logs/
│   ├── plots/
│   └── reports/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
│
├── config/                         # Configuration files
│   ├── datasets.yaml
│   ├── models.yaml
│   └── experiments.yaml
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_experiments.py
│
├── docs/
│   ├── guide.md
│   └── api.md
│
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```
