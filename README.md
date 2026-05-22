# Options Lab — Black-Scholes Pricer

Pricer d'options base sur le modele Black-Scholes avec visualisation interactive des Greeks.

## Fonctionnalites

- Pricing d'options europeennes (Call / Put) via Black-Scholes
- Calcul et visualisation des Greeks (Delta, Gamma, Theta, Vega, Rho)
- Surfaces de volatilite et analyse de sensibilite
- Strategies multi-jambes (spreads, straddles, combos)
- Graphiques SVG generes dynamiquement (zero dependance graphique externe)

## Stack technique

- **Python** - Streamlit, NumPy, Pandas, SciPy
- **Modele** - Black-Scholes (scipy.stats, scipy.optimize)
- **Visualisation** - SVG pur genere dynamiquement

## Installation et lancement

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app3.py
```
