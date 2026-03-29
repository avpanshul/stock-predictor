# Stock Price Movement Predictor

An ANN-based binary classifier that predicts whether a stock will move **UP or DOWN** the next trading day, using 13 technical indicators and a full backtesting engine.

## Live Demo
> Run locally with `streamlit run app.py`

## Results

| Metric | Baseline (Logistic Regression) | ANN Model |
|---|---|---|
| Accuracy | 48.61% | 53.67% |
| Precision | 62.07% | 54.78% |
| Recall | 40.91% | 92.99% |
| **F1 Score** | **49.32%** | **68.95%** |

### Backtest Results (AAPL, 2024)
| Metric | Value |
|---|---|
| Sharpe Ratio | 1.51 |
| Max Drawdown | -13.34% |
| Win Rate | 56.70% |
| Strategy Return | 34.14% |
| Market Return | 35.59% |



## Tech Stack

- **Data** — yfinance, pandas
- **Indicators** — pandas-ta (RSI, MACD, EMA, Bollinger Bands, ATR, Stochastic, Williams %R, ROC)
- **ML** — TensorFlow/Keras, scikit-learn
- **App** — Streamlit
- **Visualisation** — matplotlib

## Model Architecture

Input (13 features)
→ Dense(128, ReLU) + BatchNorm + Dropout(0.3)
→ Dense(64, ReLU)  + BatchNorm + Dropout(0.25)
→ Dense(32, ReLU)  + Dropout(0.2)
→ Dense(1, Sigmoid)

## How to Run Locally
bash
# Clone the repo
git clone https://github.com/avpanshul/stock-predictor.git
cd stock-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the notebook (to train the model)
jupyter notebook stock_predictor.ipynb

# Run the app
streamlit run app.py


## Project Structure

stock-predictor/
│
├── stock_predictor.ipynb     
├── app.py                    
├── requirements.txt          
├── README.md                 
│
├── equity_curve.png          
├── training_curves.png       
└── baseline_confusion_matrix.png

## Features

- Downloads real stock data via Yahoo Finance
- Computes 13 technical indicators automatically
- Trains ANN on 4 stocks simultaneously (4,900 samples)
- Compares against Logistic Regression baseline
- Full backtesting engine with Sharpe ratio and drawdown
- Interactive Streamlit app — enter any ticker and get a prediction

## Disclaimer

*This project is for educational purposes only. Do not use for real trading decisions.*

