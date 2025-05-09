# inavvi

Runtime Environment: Python 3.10+
GPU Acceleration: CUDA 12.x on A100
Data Processing: RAPIDS (cuDF, cuML)
ML Frameworks: PyTorch, Transformers, XGBoost
LLM Provider: OpenRouter API
API Client: FastAPI with WebSockets
Storage: Redis, TimeScaleDB
Configuration: Pydantic + python-dotenv
Monitoring: Custom dashboard with Plotly Dash

Directory Structure
trading_system/
├── .env                        # Environment variables
├── config.yaml                 # System configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── data/                       # Data storage
│   ├── raw/                    # Raw data dumps
│   └── processed/              # Processed datasets
├── logs/                       # Log files
├── models/                     # Saved ML models
├── src/                        # Source code
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config/                 # Configuration modules
│   │   ├── __init__.py
│   │   └── settings.py         # App settings/configuration
│   ├── api/                    # API layer
│   │   ├── __init__.py
│   │   ├── endpoints.py        # FastAPI routes
│   │   └── websocket.py        # WebSocket handlers
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── data_pipeline.py    # Data collection pipeline
│   │   ├── screening.py        # Stock screening logic
│   │   ├── position_monitor.py # Position monitoring
│   │   └── trade_execution.py  # Trade execution logic
│   ├── data_sources/           # Data source connectors
│   │   ├── __init__.py
│   │   ├── polygon.py          # Polygon REST/WebSocket API
│   │   ├── alpha_vantage.py    # Alpha Vantage API
│   │   └── yahoo_finance.py    # Yahoo Finance API
│   ├── models/                 # ML model definitions
│   │   ├── __init__.py
│   │   ├── pattern_recognition.py
│   │   ├── sentiment.py
│   │   └── exit_optimization.py
│   ├── training/               # Model training scripts
│   │   ├── __init__.py
│   │   ├── data_preparation.py
│   │   └── train_models.py
│   ├── llm/                    # LLM integration
│   │   ├── __init__.py
│   │   ├── router.py           # OpenRouter connection
│   │   ├── prompts.py          # LLM prompts
│   │   └── parsing.py          # Response parsing
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py          # Logging setup
│   │   └── redis_client.py     # Redis connection
│   └── dashboard/              # Monitoring dashboard
│       ├── __init__.py
│       ├── app.py              # Dash application
│       └── components.py       # Dashboard components
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_data_sources.py
    ├── test_models.py
    └── test_trading_logic.py