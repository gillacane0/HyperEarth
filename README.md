**HyperEarth**

HyperEarth is a framework with the objective of testing and comparing different deep-learning model on differents datasets dor the Earth Observation, specifically for Hyper Spectral type of data.

Intended structure for now:


hyperspectral-benchmarking/
│
├── src/                          
│   ├── data/                     
│   │   ├── loaders.py            # Loading dataset script 
│   │   ├── preprocessing.py      # Preprocessing data
│   │   └── utils.py              
│   │
│   ├── models/                   # Models definitions
│   │   ├── baselines/            
│   │   │   ├── svm.py
│   │   │   ├── random_forest.py
│   │   │   └── knn.py
│   │   │
│   │   ├── cnn/                 
│   │   │   ├── cnn_1d.py
│   │   │   ├── cnn_2d.py
│   │   │   └── cnn_3d.py
│   │   │
│   │   ├── transformers/         
│   │   │   ├── vit.py            
│   │   │   └── spectralformer.py
│   │   │
│   │   ├── hybrids/              
│   │   │   ├── cnn_lstm.py
│   │   │   └── cnn_transformer.py
│   │   │
│   │   └── utils.py              
│   │
│   ├── experiments/              
│   │   ├── train.py              
│   │   ├── evaluate.py          
│   │   └── benchmark.py          
│   │
│   └── utils/                    
│       ├── logging.py            
│       └── visualization.py      
│
├── data/                         # Dataset and inputs data
│   ├── raw/                      
│   ├── processed/                
│   └── external/                 
│
├── results/                      
│   ├── logs/                     
│   ├── plots/                    
│   └── reports/                 
│
├── notebooks/                    
│   ├── exploration.ipynb         
│   └── results_analysis.ipynb    
│
├── config/                       
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
