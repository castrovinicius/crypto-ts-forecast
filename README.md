# Bitcoin Price Forecast API

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Prophet](https://img.shields.io/badge/Prophet-1.2.1-orange.svg)](https://facebook.github.io/prophet/)

## Overview

A Bitcoin price forecasting system built with Kedro for ML pipeline orchestration, Prophet for time series forecasting, and FastAPI for serving predictions via REST API.

### Features

- **Automated Data Ingestion**: Fetches historical Bitcoin data from Binance API
- **Prophet Forecasting**: Uses Facebook Prophet for time series predictions with seasonality modeling
- **Kedro Pipelines**: Well-organized, reproducible ML pipelines with data lineage
- **REST API**: FastAPI-based API with automatic documentation and validation
- **Configurable Predictions**: Forecasts up to 365 days ahead with confidence intervals

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Binance API    │────▶│  Kedro Pipeline │────▶│  Prophet Model │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client/App    │◀────│   FastAPI       │◀────│   Predictions  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd crypto-ts-forecast

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Docker

You can also build and run the application using Docker (with `uv` for fast builds):

```bash
# Build the image
docker build -t crypto-ts-forecast .

# Run the container
docker run --rm -p 8000:8000 --name crypto-api crypto-ts-forecast
```

The API will be available at `http://localhost:8000/docs`

### Pipeline Execution

Execute the complete ML pipeline to train the model with historical Bitcoin data:

```bash
kedro run
```

Pipeline stages:
1. Data ingestion from Binance API (2 years of daily OHLCV data)
2. Data processing and transformation for Prophet format
3. Model training with seasonality components
4. Forecast generation for configurable time horizon

### API Server

Start the FastAPI server:

```bash
# Direct module execution
python -m crypto_ts_forecast.api.main

# Entry point command
crypto-forecast-api
```

The API will be available at `http://localhost:8000`

### API Documentation

Open your browser and go to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
```

### Get Current Bitcoin Price
```bash
GET /api/v1/price/current?symbol=BTCUSDT
```

### Get Forecast
```bash
GET /api/v1/forecast?days_ahead=30
```

### Generate New Forecast (with optional retraining)
```bash
POST /api/v1/forecast
Content-Type: application/json

{
    "days_ahead": 30,
    "retrain": false
}
```

### Run Pipeline
```bash
POST /api/v1/pipelines/run
Content-Type: application/json

{
    "pipeline_name": "__default__"
}
```

### Get Model Info
```bash
GET /api/v1/model/info
```

## Usage Examples

### Python Client
```python
import requests

# Get forecast
response = requests.get("http://localhost:8000/api/v1/forecast?days_ahead=7")
forecast = response.json()

for prediction in forecast["predictions"]:
    print(f"{prediction['date']}: ${prediction['predicted_price']:,.2f}")
```

### Command Line
```bash
# Get 7-day forecast
curl "http://localhost:8000/api/v1/forecast?days_ahead=7"

# Retrain model and get forecast
curl -X POST "http://localhost:8000/api/v1/forecast" \
     -H "Content-Type: application/json" \
     -d '{"days_ahead": 30, "retrain": true}'
```

## Kedro Pipelines

### Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `data_ingestion` | Fetches Bitcoin data from Binance API |
| `data_processing` | Transforms data to Prophet format |
| `model_training` | Trains and evaluates Prophet model |
| `inference` | Generates future predictions |
| `__default__` | Runs all pipelines in sequence |

### Run Specific Pipeline
```bash
kedro run --pipeline data_ingestion
kedro run --pipeline model_training
```

### Visualize Pipeline
```bash
kedro viz
```

## Configuration

### Pipeline Parameters

Configurable via `conf/base/parameters.yml`:

```yaml
binance:
  symbol: "BTCUSDT"
  interval: "1d"
  years_of_data: 2

prophet:
  price_column: "close"
  seasonality_mode: "multiplicative"
  yearly_seasonality: true
  weekly_seasonality: true
  changepoint_prior_scale: 0.5
  seasonality_prior_scale: 10.0
  changepoint_range: 0.9
  test_size_days: 30

forecast:
  days_ahead: 30
```

### Data Management

Data catalog configured in `conf/base/catalog.yml`. Datasets are stored as Parquet files with gzip compression. Trained models are persisted as pickle files.

## Project Structure

```
crypto-ts-forecast/
├── conf/                      # Configuration files
│   ├── base/
│   │   ├── catalog.yml        # Data catalog
│   │   └── parameters.yml     # Parameters
│   └── local/
│       └── credentials.yml    # Credentials (gitignored)
├── data/                      # Data storage
│   ├── 01_raw/                # Raw data from Binance
│   ├── 02_intermediate/       # Validated data
│   ├── 03_primary/            # Prophet base dataset
│   ├── 04_feature/            # Feature-enhanced dataset
│   ├── 05_model_input/        # Train/test splits
│   ├── 06_models/             # Trained Prophet model
│   ├── 07_model_output/       # Predictions
│   └── 08_reporting/          # Reports and metrics
├── src/crypto_ts_forecast/
│   ├── api/                   # FastAPI application
│   │   ├── app.py             # App factory and routes
│   │   ├── schemas.py         # Pydantic models
│   │   ├── services.py        # Business logic
│   │   └── main.py            # Entry point
│   └── pipelines/
│       ├── data_ingestion/    # Binance data fetching
│       ├── data_processing/   # Data transformation
│       ├── model_training/    # Prophet training
│       └── inference/         # Forecast generation
└── tests/                     # Unit tests
```

## Development

### Run Tests
```bash
pytest
```

### Lint Code
```bash
ruff check .
```

### Format Code
```bash
ruff format .
```

## Requirements

- Python 3.10+
- Kedro 1.1.1
- Prophet 1.2.1
- FastAPI 0.104+
- See `requirements.txt` for full list

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This is not financial advice.** The predictions generated by this system are for educational and informational purposes only. Cryptocurrency markets are highly volatile and past performance does not guarantee future results. Users should conduct their own research before making investment decisions.
