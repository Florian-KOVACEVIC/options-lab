# Options Lab — Black-Scholes Pricer

**[Live Demo](https://options-lab-florian-kovacevic.streamlit.app/)**

Interactive options pricing tool built on the Black-Scholes model, with full Greeks visualization and multi-leg strategy builder.

Built as a personal project to implement derivatives pricing theory from scratch.

## Features

### Options Pricing
- European option pricing (Call/Put) using the Black-Scholes closed-form solution
- Implied volatility solver via Brent's method (scipy.optimize)
- Moneyness analysis (ITM / ATM / OTM) with visual indicators

### Greeks Analysis
- Full Greeks computation: Delta, Gamma, Theta, Vega, Rho
- Sensitivity charts showing Greeks evolution across spot price and time
- Greeks surface visualization for strike/maturity combinations

### Strategy Builder
- Multi-leg strategy construction (spreads, straddles, strangles, butterflies, custom combos)
- P&L payoff diagrams at expiration
- Combined Greeks aggregation across all legs
- Pre-built strategy templates

### Technical Choices
- All charts rendered as **pure dynamically-generated SVG** — zero dependency on matplotlib or plotly
- Lightweight and fast, works on any platform without additional graphical libraries

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python |
| **Framework** | Streamlit |
| **Pricing Model** | Black-Scholes (implemented from scratch) |
| **Math** | NumPy, SciPy (stats, optimize) |
| **Visualization** | Pure SVG generation |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Florian-KOVACEVIC/options-lab.git
cd options-lab

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Launch the app
python -m streamlit run app3.py
```

The app opens at `http://localhost:8501`.

## Concepts Implemented

| Concept | Description |
|---------|-------------|
| **Black-Scholes** | Closed-form European option pricing under log-normal assumptions |
| **Greeks** | First and second-order sensitivities (Delta, Gamma, Theta, Vega, Rho) |
| **Implied Volatility** | Numerical inversion of BS formula using Brent's root-finding |
| **Payoff Diagrams** | P&L profiles for single and multi-leg strategies at expiration |

## Disclaimer

This tool is for educational and personal use only. It does not constitute financial advice or trading recommendations.
