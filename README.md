# market-regime-prediction

# Regime-Switching Market Timing Strategy

A cross-validated implementation of a Gaussian Mixture Model (GMM) regime detection system for S&P 500 market timing.

## Features
- **Regime Detection**: 2-component GMM clustering on rolling volatility & momentum
- **Parameter Optimization**: Walk-forward cross-validation across multiple temporal windows
- **Risk Management**: Dynamic allocation between market index and risk-free rate

## Usage
```bash
pip install -r requirements.txt
python main.py
