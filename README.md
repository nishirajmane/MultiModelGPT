# 🧠 NeuroQuantix: Advanced Financial AI Research

A comprehensive research and development experiment implementing advanced AI systems for financial market prediction using real market data, technical indicators, and sophisticated neural architectures.

## 📊 Project Overview

NeuroQuantix combines advanced time-series analysis with real market data to predict cryptocurrency and stock price movements. The system features multiple model architectures including a basic multimodal system and an advanced optimized model with enhanced features.

## 🏗️ Architecture

### Core Components:

#### 1. Basic Multimodal System (research_experiment.py)
- **TimeSeriesTransformer**
  - Input: OHLCV data (shape: [batch_size, 30, 5])
  - Linear projection to 64-dim embeddings
  - 2-layer TransformerEncoder with 4 attention heads
  - Output: Time-series embeddings [batch_size, 64]

- **FinBERTWrapper**
  - Uses pretrained "yiyanghkust/finbert-tone" model
  - Encodes financial news headlines into sentiment embeddings
  - Output: 64-dim sentiment vectors [batch_size, 64]

- **Cross-Attention Fusion**
  - MultiHeadAttention layer for modality fusion
  - Time-series embeddings as query, news embeddings as key/value
  - Output: Fused representation [batch_size, 64]

#### 2. Advanced Optimized System (optimized_market_encoder.py)
- **Enhanced Input Layer**
  - 37 real market features including OHLCV, technical indicators, and volatility measures
  - Multi-scale feature embedding with 256-dimensional hidden layers
  - Robust NaN handling and data validation

- **Advanced Multi-Scale Convolution**
  - Kernel sizes: [3, 7, 15, 31] for capturing different temporal patterns
  - Temporal convolutional residual blocks with volatility embedding
  - Learnable volatility positional encoding

- **Regime-Aware Transformer**
  - 6-layer transformer with 16 attention heads
  - Volatility regime detection and embedding
  - Cross-time attention with regime-aware mechanisms

- **Multi-Task Prediction Heads**
  - Direction prediction (3-class: Down, Sideways, Up)
  - Return prediction with confidence scoring
  - Volatility forecasting
  - Market regime classification

## 📈 Results

### Basic Model Performance:
- **Training Accuracy:** 37.50%
- **Strategy Sharpe Ratio:** 0.1212
- **Market Sharpe Ratio:** 0.2004
- **Model Parameters:** 110M+ (including FinBERT)

### Advanced Model Features:
- **Real Market Data Integration:** Live BTC-USD data via yfinance
- **37 Technical Features:** RSI, MACD, Bollinger Bands, ADX, CCI, Williams %R, Stochastic
- **Volatility Regime Detection:** Low/Medium/High volatility classification
- **Enhanced Loss Functions:** Focal Loss, Weighted Cross-Entropy, Brier Score
- **Comprehensive Metrics:** Per-class accuracy, precision, recall, F1-score
- **Robust Error Handling:** NaN handling, shape validation, fallback mechanisms

## 🚀 Setup & Installation

### Prerequisites:
```bash
pip install -r requirements.txt
```

### Dependencies:
- PyTorch >= 1.13.0
- Transformers >= 4.20.0
- yfinance >= 0.2.0
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.1.0
- seaborn >= 0.11.0
- numpy >= 1.21.0

### Running the Experiments:

#### Basic Multimodal System:
```bash
python research_experiment.py
```

#### Advanced Optimized System:
```bash
python NeuroQuantix_Advanced/optimized_market_encoder.py
```

## 📁 Project Structure

```
MultiModelGPT/
├── research_experiment.py                    # Basic multimodal experiment
├── NeuroQuantix_Advanced/
│   ├── optimized_market_encoder.py          # Advanced optimized model
│   ├── advanced_market_encoder.py           # Enhanced market encoder
│   └── requirements.txt                     # Advanced dependencies
├── requirements.txt                         # Basic dependencies
├── README.md                               # This file
└── .gitignore                             # Git ignore file
```

## 🔬 Research Methodology

### Data Processing:
- **Real Market Data:** Live BTC-USD data via yfinance API
- **Feature Engineering:** 37 technical indicators and market features
- **Normalization:** Robust scaling with NaN handling
- **Labels:** 3-class direction classification (Down/Sideways/Up)
- **Sequence Length:** 24 time steps with 37 features

### Advanced Training:
- **Optimizer:** Adam with learning rate 0.0005
- **Loss Functions:** Multi-objective with focal loss and weighted cross-entropy
- **Epochs:** 100-150 with early stopping
- **Batch Size:** Dynamic based on data size
- **Class Balancing:** Dynamic class weight calculation

### Enhanced Evaluation:
- **Per-Class Metrics:** Precision, recall, F1-score for each direction
- **Financial Metrics:** Sharpe ratio, max drawdown, cumulative returns
- **Attention Analysis:** Regime-specific attention weight analysis
- **Confidence Calibration:** Brier score for prediction confidence

## 🎯 Key Features

### Basic System:
- **Multimodal Fusion:** Combines technical and sentiment data
- **Transformer Architecture:** State-of-the-art sequence modeling
- **Financial Focus:** Specialized for market prediction

### Advanced System:
- **Real-Time Data:** Live market data integration
- **Comprehensive Features:** 37 technical and market indicators
- **Regime Awareness:** Volatility-based market state detection
- **Robust Architecture:** Error handling and NaN management
- **Multi-Task Learning:** Direction, return, confidence, and volatility prediction
- **Attention Diagnostics:** Detailed attention weight analysis
- **Production Ready:** Comprehensive error handling and validation

## 🚀 Recent Improvements

### Latest Updates (v2.0):
- ✅ **Comprehensive Error Handling:** Robust NaN handling throughout the pipeline
- ✅ **Enhanced Metrics Calculation:** Safe tensor operations and fallback mechanisms
- ✅ **Improved Visualization:** Error-resistant plotting with fallback displays
- ✅ **Attention Analysis:** Regime-specific attention weight analysis
- ✅ **Production Robustness:** Extensive try-catch blocks and validation
- ✅ **Real Market Integration:** Live BTC-USD data with 5000+ data points
- ✅ **Advanced Architecture:** 6-layer transformer with 16 attention heads
- ✅ **Multi-Objective Training:** Focal loss, weighted cross-entropy, and Brier score

## 📊 Visualization Output

The advanced system generates comprehensive visualizations including:
- Real-time BTC price charts with technical indicators
- Training loss progression with validation curves
- Per-class performance breakdown (Down/Sideways/Up)
- Attention heatmaps for different volatility regimes
- Performance metrics comparison
- Volume and volatility analysis

## 🤝 Contributing

This is an active research project. Feel free to:
- Experiment with different architectures and features
- Test on different assets and time periods
- Improve the backtesting methodology
- Add new technical indicators and features
- Enhance the error handling and robustness
- Optimize the model performance

## 📄 License

This project is for research purposes. Please ensure compliance with relevant financial regulations when using for actual trading.

---

**Note:** This is a research experiment and should not be used for actual trading without proper validation and risk management. The advanced system includes comprehensive error handling and is designed for research and educational purposes. 