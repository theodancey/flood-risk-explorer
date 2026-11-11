# Flood Risk Explorer

An actuarial-style look at flood insurance risk built on FEMA's public National Flood Insurance Program (NFIP) claims and policy data from OpenFEMA. It models claim **frequency** (Poisson / negative binomial GLMs) and claim **severity** (Gamma GLM), compares them with gradient boosting (XGBoost), and explains what drives risk. The final product is a Streamlit app: enter a ZIP code or county and see area-level flood risk, expected claim cost, and the reasons behind them.

**Status: in progress**

> **Disclaimer:** All results are **area-level** estimates built from aggregated public data. They are not predictions, quotes, or risk ratings for any individual property.

## Data
_Coming soon: OpenFEMA NFIP claims and policies datasets, how they're downloaded, and how they're cleaned._

## Methods
_Coming soon: frequency and severity GLMs, the XGBoost comparison, and model explanations (SHAP)._

## Results
_Coming soon._

## How to Run
_Full instructions coming soon. Environment setup so far (Windows, PowerShell):_

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\check_setup.py
```
