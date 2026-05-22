"""
Options Lab — Black-Scholes Pricer
Graphiques : SVG pur généré dynamiquement (zéro matplotlib, zéro plotly)
Compatible Windows/Mac/Linux sans installation supplémentaire
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import warnings, html
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Options Lab", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root{
  --bg:#09090b;--s1:#111113;--s2:#18181b;--s3:#1f2024;
  --b1:#27272a;--b2:#3f3f46;
  --g:#22c55e;--g2:#16a34a;
  --b:#3b82f6;--b2c:#2563eb;
  --o:#f59e0b;--r:#ef4444;--p:#a78bfa;
  --t:#fafafa;--t2:#a1a1aa;--t3:#52525b;--t4:#27272a;
  --rad:10px;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background:var(--bg)!important;color:var(--t);}
.main .block-container{padding:22px 30px 48px;max-width:1580px;}
#MainMenu,footer,header{visibility:hidden;}
/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;
  padding-bottom:18px;border-bottom:1px solid var(--b1);margin-bottom:20px;}
.hdr-l{display:flex;align-items:center;gap:12px;}
.hdr-ico{width:34px;height:34px;border-radius:8px;
  background:linear-gradient(135deg,#3b82f6,#8b5cf6);
  display:flex;align-items:center;justify-content:center;color:#fff;font-size:15px;font-weight:700;}
.hdr-t{font-size:1.05rem;font-weight:700;letter-spacing:-.3px;}
.hdr-s{font-size:.68rem;color:var(--t3);margin-top:1px;}
.live{font-size:.64rem;font-weight:600;padding:4px 10px;border-radius:20px;
  background:rgba(34,197,94,.1);color:var(--g);border:1px solid rgba(34,197,94,.2);
  font-family:'DM Mono',monospace;}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--s1);border:1px solid var(--b1);border-radius:9px;padding:3px;gap:2px;}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif!important;font-size:.77rem;font-weight:500;
  color:var(--t3);border-radius:7px;padding:7px 18px;}
.stTabs [aria-selected="true"]{background:var(--s3)!important;color:var(--t)!important;box-shadow:0 1px 4px rgba(0,0,0,.4);}
/* Cards */
.card{background:var(--s1);border:1px solid var(--b1);border-radius:var(--rad);padding:18px;}
.ct{font-size:.60rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--t3);margin-bottom:12px;}
/* Price hero */
.ph{background:var(--s1);border:1px solid var(--b1);border-radius:var(--rad);
  padding:20px 24px;display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:14px;}
.ph-ey{font-size:.60rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--t3);margin-bottom:6px;}
.ph-row{display:flex;align-items:baseline;gap:5px;}
.ph-cur{font-size:1.1rem;color:var(--t3);font-family:'DM Mono',monospace;}
.ph-val{font-size:2.6rem;font-weight:700;letter-spacing:-2px;font-family:'DM Mono',monospace;line-height:1;}
.ph-sub{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--t3);margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;}
.ph-badge{font-size:.70rem;font-weight:600;padding:5px 14px;border-radius:6px;align-self:flex-start;}
.ph-c{background:rgba(34,197,94,.1);color:var(--g);border:1px solid rgba(34,197,94,.2);}
.ph-p{background:rgba(167,139,250,.1);color:var(--p);border:1px solid rgba(167,139,250,.2);}
/* Greek card */
.gc{background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:13px;position:relative;overflow:hidden;}
.gc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--acc,#3b82f6);}
.gc-sym{font-family:'DM Mono',monospace;font-size:1.35rem;font-weight:500;color:var(--acc,#3b82f6);line-height:1;}
.gc-nm{font-size:.58rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--t3);margin-top:2px;}
.gc-v{font-family:'DM Mono',monospace;font-size:1.0rem;font-weight:600;margin-top:9px;}
.gc-d{font-size:.64rem;color:var(--t3);margin-top:5px;line-height:1.5;}
/* Signals */
.sig{display:flex;align-items:flex-start;gap:9px;padding:10px 13px;
  border-radius:8px;border:1px solid;margin:4px 0;font-size:.73rem;line-height:1.65;}
.dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:6px;}
.sg{background:rgba(34,197,94,.06);border-color:rgba(34,197,94,.2);color:#86efac;}
.sr{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.2);color:#fca5a5;}
.sb{background:rgba(59,130,246,.06);border-color:rgba(59,130,246,.2);color:#93c5fd;}
.so{background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.2);color:#fcd34d;}
.dg{background:#22c55e;}.dr{background:#ef4444;}.db{background:#3b82f6;}.do{background:#f59e0b;}
/* Sec header */
.sh{font-size:.60rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--t3);padding:14px 0 8px;border-bottom:1px solid var(--b1);margin-bottom:11px;}
/* Field label */
.fl{font-size:.69rem;font-weight:500;color:var(--t3);margin-bottom:4px;margin-top:10px;}
/* Mat badge */
.mb{font-family:'DM Mono',monospace;font-size:.73rem;color:var(--b);
  background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.18);
  border-radius:6px;padding:5px 12px;margin:5px 0 10px;text-align:center;display:block;}
/* Moneyness */
.mpill{font-size:.64rem;font-weight:600;padding:2px 8px;border-radius:4px;display:inline-block;}
.matm{background:rgba(245,158,11,.1);color:var(--o);}
.mitm{background:rgba(34,197,94,.1);color:var(--g);}
.motm{background:rgba(239,68,68,.1);color:var(--r);}
/* IV */
.ivr{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.2);border-radius:7px;
  padding:10px 14px;margin-top:8px;font-family:'DM Mono',monospace;font-size:.85rem;color:var(--g);font-weight:600;}
.ivf{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);border-radius:7px;
  padding:10px 14px;margin-top:8px;font-size:.77rem;color:var(--r);}
/* Strat card */
.sc{background:var(--s1);border:1px solid var(--b1);border-radius:9px;padding:13px 14px;margin-bottom:7px;}
.sc-sel{border-color:#3b82f6!important;background:var(--s2)!important;}
.sc-nm{font-size:.82rem;font-weight:600;margin-bottom:5px;}
.sc-s{font-size:.66rem;color:var(--t3);line-height:1.5;}
.tag{display:inline-block;font-size:.58rem;font-weight:600;padding:2px 6px;border-radius:4px;margin:1px 2px;}
.tby{background:rgba(34,197,94,.12);color:#22c55e;}
.tse{background:rgba(239,68,68,.12);color:#ef4444;}
.tca{background:rgba(59,130,246,.12);color:#3b82f6;}
.tpu{background:rgba(167,139,250,.12);color:#a78bfa;}
/* Total bar */
.tbar{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0;}
.tbi{background:var(--s1);border:1px solid var(--b1);border-radius:8px;padding:12px 14px;text-align:center;}
.tbl{font-size:.58rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:1px;}
.tbv{font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:700;margin-top:4px;}
/* Leg card */
.lc{border-radius:9px;padding:15px 18px;margin-bottom:10px;border:1px solid;}
.lb{background:rgba(34,197,94,.04);border-color:rgba(34,197,94,.18);}
.ls{background:rgba(239,68,68,.04);border-color:rgba(239,68,68,.18);}
.lo{background:var(--s1);border-color:var(--b1);opacity:.5;}
.lh{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
.ln{font-size:.80rem;font-weight:700;}
/* Pinfo row */
.prow{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;}
.pi{flex:1;min-width:105px;background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:10px 12px;}
.pil{font-size:.57rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:1px;}
.piv{font-size:.80rem;font-weight:600;margin-top:3px;}
/* Inputs */
div[data-testid="stNumberInput"] input{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;font-family:'DM Mono',monospace!important;font-size:.83rem!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;}
div[data-testid="stTextInput"] input{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;}
.stSlider [data-baseweb="slider"]{padding:4px 0;}
[data-testid="stSlider"] label{color:var(--t3)!important;font-size:.75rem!important;}
.stRadio>label{color:var(--t2)!important;font-size:.75rem!important;}
.stRadio [data-testid="stMarkdownContainer"] p{font-size:.75rem!important;}
.stButton>button{background:var(--s2)!important;border:1px solid var(--b1)!important;color:var(--t2)!important;
  font-family:'Inter',sans-serif!important;font-size:.73rem!important;font-weight:500!important;
  border-radius:7px!important;padding:6px 14px!important;width:100%!important;transition:all .15s!important;}
.stButton>button:hover{border-color:#3b82f6!important;color:var(--t)!important;}
.stCheckbox [data-testid="stMarkdownContainer"] p{font-size:.75rem!important;color:var(--t2)!important;}
.streamlit-expanderHeader{font-family:'Inter',sans-serif!important;font-size:.73rem!important;
  color:var(--t3)!important;background:var(--s1)!important;border:1px solid var(--b1)!important;border-radius:8px!important;}
.streamlit-expanderContent{background:var(--s1)!important;border:1px solid var(--b1)!important;border-top:none!important;}
label,.stMarkdown p{color:var(--t2)!important;font-size:.78rem!important;}
hr{border-color:var(--b1);margin:14px 0;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:#3f3f46;border-radius:4px;}
/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:var(--s1)!important;
  border-right:1px solid var(--b1)!important;
  min-width:280px!important;
  max-width:320px!important;
}
section[data-testid="stSidebar"] .block-container{padding:18px 16px 32px!important;}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p{color:var(--t2)!important;font-size:.76rem!important;}
section[data-testid="stSidebar"] [data-testid="stSlider"] label{color:var(--t3)!important;font-size:.73rem!important;}
section[data-testid="stSidebar"] hr{border-color:var(--b1);margin:10px 0;}
.sb-title{font-size:.60rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--t3);padding:10px 0 6px;border-bottom:1px solid var(--b1);margin-bottom:8px;}
.sb-badge{display:inline-flex;align-items:center;gap:6px;font-size:.68rem;font-weight:600;
  padding:4px 10px;border-radius:6px;margin-bottom:10px;}
.sb-call{background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);}
.sb-put{background:rgba(167,139,250,.1);color:#a78bfa;border:1px solid rgba(167,139,250,.2);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MATH ENGINE
# ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, q=0.0, otype="call"):
    if T <= 1e-9 or sigma <= 1e-9:
        if otype == "call": return max(S*np.exp(-q*T) - K*np.exp(-r*T), 0)
        return max(K*np.exp(-r*T) - S*np.exp(-q*T), 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if otype == "call": return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, q=0.0, otype="call"):
    if T <= 1e-9 or sigma <= 1e-9:
        return dict(delta=0,gamma=0,theta=0,vega=0,rho=0,vanna=0)
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    nd1=norm.pdf(d1); sign=1 if otype=="call" else -1
    return dict(
        delta = sign*np.exp(-q*T)*norm.cdf(sign*d1),
        gamma = np.exp(-q*T)*nd1/(S*sigma*np.sqrt(T)),
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 -sign*r*K*np.exp(-r*T)*norm.cdf(sign*d2)
                 +sign*q*S*np.exp(-q*T)*norm.cdf(sign*d1))/365,
        vega  = S*np.exp(-q*T)*nd1*np.sqrt(T)/100,
        rho   = sign*K*T*np.exp(-r*T)*norm.cdf(sign*d2)/100,
        vanna = -np.exp(-q*T)*nd1*d2/sigma,
    )

def implied_vol(mkt, S, K, T, r, q=0.0, otype="call"):
    try:
        f = lambda s: bs_price(S,K,T,r,s,q,otype)-mkt
        if f(1e-4)*f(9.9)>=0: return np.nan
        return brentq(f,1e-4,9.9)
    except Exception: return np.nan

def mat_from_ymd(y,m,d): return max(y+m/12+d/365, 1/365)

def fmt_mat(T):
    y=int(T); rm=T-y; mo=int(rm*12); rm2=rm-mo/12; dys=int(round(rm2*365))
    p=[]
    if y:   p.append(f"{y}a")
    if mo:  p.append(f"{mo}m")
    if dys: p.append(f"{dys}j")
    return " ".join(p) or "<1j"

# ─────────────────────────────────────────────────────────────
#  SVG CHART ENGINE — no matplotlib needed
# ─────────────────────────────────────────────────────────────
BG_SVG  = "#09090b"
S1_SVG  = "#111113"
GR_SVG  = "#27272a"
TC_SVG  = "#52525b"
T2_SVG  = "#71717a"

def _norm_series(xs, ys, W, H, PAD_L=52, PAD_R=16, PAD_T=16, PAD_B=36):
    """Normalise data → SVG pixel coords"""
    xmin,xmax = min(xs),max(xs)
    ymin,ymax = min(ys),max(ys)
    if xmax==xmin: xmax=xmin+1
    if ymax==ymin: ymin,ymax=ymin-1,ymax+1
    pw = W - PAD_L - PAD_R
    ph = H - PAD_T - PAD_B
    px = [PAD_L + (x-xmin)/(xmax-xmin)*pw for x in xs]
    py = [PAD_T + ph - (y-ymin)/(ymax-ymin)*ph for y in ys]
    return px, py, xmin, xmax, ymin, ymax, PAD_L, PAD_R, PAD_T, PAD_B, pw, ph

def svg_chart(
    series_list,           # [{"x":[], "y":[], "color":"#hex", "label":"", "width":2, "dash":False, "fill":False, "fill_pos_neg":False}]
    W=700, H=260,
    title="",
    xlabel="", ylabel="",
    vlines=None,           # [{"x": val, "color":"#hex", "label":"", "dash":True}]
    hline_zero=False,
    show_dot=None,         # {"x":val,"y":val,"color":"#hex"}
    PAD_L=54, PAD_R=18, PAD_T=28, PAD_B=40,
):
    pw = W - PAD_L - PAD_R
    ph = H - PAD_T - PAD_B

    # Gather all y values for global scale
    all_x = [v for s in series_list for v in s["x"]]
    all_y = [v for s in series_list for v in s["y"]]
    if not all_x: return ""
    xmin,xmax = min(all_x),max(all_x)
    ymin,ymax = min(all_y),max(all_y)
    if xmax==xmin: xmax+=1
    span = ymax-ymin
    if span < 1e-12: ymin,ymax = ymin-0.5,ymax+0.5; span=1
    ymin -= span*0.04; ymax += span*0.04; span = ymax-ymin

    def tx(v): return PAD_L + (v-xmin)/(xmax-xmin)*pw
    def ty(v): return PAD_T + ph - (v-ymin)/span*ph
    def ty_zero(): return ty(0)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'style="background:{BG_SVG};border-radius:10px;display:block">']

    # Grid lines y
    n_yticks = 5
    for i in range(n_yticks+1):
        yv = ymin + i*(ymax-ymin)/n_yticks
        yp = ty(yv)
        svg.append(f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{PAD_L+pw}" y2="{yp:.1f}" '
                   f'stroke="{GR_SVG}" stroke-width="1" opacity=".7"/>')
        label_v = f"{yv:.3f}" if abs(yv)<10 else f"{yv:.1f}"
        svg.append(f'<text x="{PAD_L-4}" y="{yp+4:.1f}" text-anchor="end" '
                   f'font-family="DM Mono,monospace" font-size="9" fill="{TC_SVG}">{label_v}</text>')

    # Grid lines x
    n_xticks = 5
    for i in range(n_xticks+1):
        xv = xmin + i*(xmax-xmin)/n_xticks
        xp = tx(xv)
        svg.append(f'<line x1="{xp:.1f}" y1="{PAD_T}" x2="{xp:.1f}" y2="{PAD_T+ph}" '
                   f'stroke="{GR_SVG}" stroke-width="1" opacity=".5"/>')
        lv = f"{xv:.0f}" if abs(xv)>=1 else f"{xv:.2f}"
        svg.append(f'<text x="{xp:.1f}" y="{PAD_T+ph+14}" text-anchor="middle" '
                   f'font-family="DM Mono,monospace" font-size="9" fill="{TC_SVG}">{lv}</text>')

    # Zero line
    if hline_zero and ymin<0<ymax:
        yz = ty(0)
        svg.append(f'<line x1="{PAD_L}" y1="{yz:.1f}" x2="{PAD_L+pw}" y2="{yz:.1f}" '
                   f'stroke="{GR_SVG}" stroke-width="1.5"/>')

    # Vertical lines
    if vlines:
        for vl in vlines:
            xp = tx(vl["x"])
            dash = 'stroke-dasharray="5,4"' if vl.get("dash",True) else ""
            svg.append(f'<line x1="{xp:.1f}" y1="{PAD_T}" x2="{xp:.1f}" y2="{PAD_T+ph}" '
                       f'stroke="{vl["color"]}" stroke-width="1.2" opacity=".7" {dash}/>')
            if vl.get("label"):
                svg.append(f'<text x="{xp+3:.1f}" y="{PAD_T+10}" font-family="DM Mono,monospace" '
                           f'font-size="8.5" fill="{vl["color"]}" opacity=".8">{vl["label"]}</text>')

    # Series
    for s in series_list:
        xs,ys = s["x"],s["y"]
        if len(xs)<2: continue
        pxs = [tx(v) for v in xs]
        pys = [ty(v) for v in ys]
        col = s.get("color","#3b82f6")
        lw  = s.get("width",2)
        dash = 'stroke-dasharray="6,4"' if s.get("dash") else ""
        poly = " ".join(f"{px:.1f},{py:.1f}" for px,py in zip(pxs,pys))

        if s.get("fill_pos_neg"):
            # Split fill: green above 0, red below
            yz = ty(0)
            poly_pos = f"{pxs[0]:.1f},{yz}"
            poly_neg = f"{pxs[0]:.1f},{yz}"
            for px,py,yv in zip(pxs,pys,ys):
                poly_pos += f" {px:.1f},{min(py,yz):.1f}"
                poly_neg += f" {px:.1f},{max(py,yz):.1f}"
            poly_pos += f" {pxs[-1]:.1f},{yz}"
            poly_neg += f" {pxs[-1]:.1f},{yz}"
            svg.append(f'<polygon points="{poly_pos}" fill="#22c55e" opacity=".13"/>')
            svg.append(f'<polygon points="{poly_neg}" fill="#ef4444" opacity=".13"/>')

        elif s.get("fill"):
            fill_col = s.get("fill_color", col)
            yz = max(PAD_T, min(PAD_T+ph, ty(0)))
            poly_fill = f"{pxs[0]:.1f},{yz} " + poly + f" {pxs[-1]:.1f},{yz}"
            svg.append(f'<polygon points="{poly_fill}" fill="{fill_col}" opacity=".09"/>')

        svg.append(f'<polyline points="{poly}" fill="none" stroke="{col}" '
                   f'stroke-width="{lw}" {dash} stroke-linejoin="round" stroke-linecap="round"/>')

    # Dot
    if show_dot:
        xp,yp = tx(show_dot["x"]), ty(show_dot["y"])
        col   = show_dot.get("color","#f59e0b")
        label = show_dot.get("label","")
        svg.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="5" fill="{col}" stroke="{BG_SVG}" stroke-width="1.5"/>')
        if label:
            svg.append(f'<text x="{xp+8:.1f}" y="{yp+4:.1f}" font-family="DM Mono,monospace" '
                       f'font-size="9" fill="{col}">{label}</text>')

    # Legend
    lx, ly = PAD_L+4, PAD_T+6
    for s in series_list:
        if not s.get("label"): continue
        svg.append(f'<rect x="{lx}" y="{ly-6}" width="16" height="3" rx="1.5" fill="{s["color"]}"/>')
        svg.append(f'<text x="{lx+20}" y="{ly+1}" font-family="Inter,sans-serif" '
                   f'font-size="9" fill="{T2_SVG}">{s["label"]}</text>')
        lx += len(s["label"])*6 + 36

    # Axes borders
    svg.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+ph}" stroke="{GR_SVG}" stroke-width="1"/>')
    svg.append(f'<line x1="{PAD_L}" y1="{PAD_T+ph}" x2="{PAD_L+pw}" y2="{PAD_T+ph}" stroke="{GR_SVG}" stroke-width="1"/>')

    # Labels
    if xlabel:
        svg.append(f'<text x="{PAD_L+pw/2:.0f}" y="{H-4}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="9.5" fill="{TC_SVG}">{xlabel}</text>')
    if ylabel:
        svg.append(f'<text transform="rotate(-90)" x="-{PAD_T+ph/2:.0f}" y="12" '
                   f'text-anchor="middle" font-family="Inter,sans-serif" font-size="9.5" fill="{TC_SVG}">{ylabel}</text>')
    if title:
        svg.append(f'<text x="{W/2:.0f}" y="{PAD_T-8}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="10" font-weight="500" fill="{T2_SVG}">{title}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def show_svg(svg_str, height=None):
    h = height or 280
    st.markdown(f'<div style="border-radius:10px;overflow:hidden;margin:4px 0">{svg_str}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────
def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def fl(t): st.markdown(f'<div class="fl">{t}</div>', unsafe_allow_html=True)

def mat_widget(pfx, dy=1, dm=0, dd=0):
    c1,c2,c3 = st.columns(3)
    with c1: y = st.number_input("Années",0,30,dy,1,key=f"{pfx}_y",help="Années entières")
    with c2: m = st.number_input("Mois",  0,11,dm,1,key=f"{pfx}_m",help="Mois supplémentaires")
    with c3: d = st.number_input("Jours", 0,30,dd,1,key=f"{pfx}_d",help="Jours supplémentaires")
    T = mat_from_ymd(y,m,d)
    st.markdown(f'<span class="mb">{fmt_mat(T)}  ·  {T:.4f} ans</span>', unsafe_allow_html=True)
    return T

def mat_inline(pfx, dy=1, dm=0, dd=0):
    c1,c2,c3 = st.columns(3)
    with c1: y=st.number_input("A",0,30,dy,1,key=f"{pfx}_y",label_visibility="collapsed",help="Années")
    with c2: m=st.number_input("M",0,11,dm,1,key=f"{pfx}_m",label_visibility="collapsed",help="Mois")
    with c3: d=st.number_input("J",0,30,dd,1,key=f"{pfx}_d",label_visibility="collapsed",help="Jours")
    T = mat_from_ymd(y,m,d)
    st.markdown(f'<span class="mb" style="font-size:.68rem">{fmt_mat(T)} · {T:.3f}a</span>',unsafe_allow_html=True)
    return T

def greek_card(sym,name,val,fmt,color,desc):
    st.markdown(f'<div class="gc" style="--acc:{color}"><div style="display:flex;align-items:flex-start;justify-content:space-between">'
                f'<span class="gc-sym">{sym}</span><span class="gc-nm">{name}</span></div>'
                f'<div class="gc-v">{val:{fmt}}</div><div class="gc-d">{desc}</div></div>',
                unsafe_allow_html=True)

def sig(sc,dc,content):
    st.markdown(f'<div class="sig {sc}"><div class="dot {dc}"></div><div>{content}</div></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  INTERPRETERS
# ─────────────────────────────────────────────────────────────
def interp(name, val):
    if name=="delta":
        if abs(val)<0.05: return "sb","db",f"<b>Delta neutre ({val:+.3f})</b> — La position ne réagit presque pas aux mouvements du marché."
        elif val>=0.5:    return "sg","dg",f"<b>Delta haussier fort ({val:+.3f})</b> — Vous gagnez quand le prix monte. Delta ≈ probabilité d'expirer dans la monnaie."
        elif val>0:       return "sg","dg",f"<b>Delta haussier ({val:+.3f})</b> — Légère exposition à la hausse."
        elif val<=-0.5:   return "sr","dr",f"<b>Delta baissier fort ({val:+.3f})</b> — Vous gagnez quand le prix baisse."
        else:             return "sr","dr",f"<b>Delta baissier ({val:+.3f})</b> — Légère exposition à la baisse."
    elif name=="gamma":
        if abs(val)<0.001: return "sb","db",f"<b>Gamma faible ({val:.5f})</b> — Votre delta change peu. Position stable et prévisible."
        elif val>0:        return "sg","dg",f"<b>Gamma positif ({val:.5f})</b> — Les gros mouvements jouent en votre faveur. Contrepartie : vous perdez de la valeur chaque jour (Theta négatif)."
        else:              return "sr","dr",f"<b>Gamma négatif ({val:.5f})</b> — Les gros mouvements vous sont défavorables. Contrepartie : vous encaissez de la valeur chaque jour (Theta positif)."
    elif name=="theta":
        if abs(val)<0.0005: return "sb","db",f"<b>Theta neutre ({val:+.4f} €/j)</b> — Le temps ne vous impacte pas significativement."
        elif val<0:         return "sr","dr",f"<b>Theta négatif ({val:+.4f} €/j)</b> — Chaque jour qui passe vous coûte {abs(val):.4f} €. Le temps est votre ennemi."
        else:               return "sg","dg",f"<b>Theta positif ({val:+.4f} €/j)</b> — Chaque jour qui passe vous rapporte {val:.4f} €. Le temps est votre allié."
    elif name=="vega":
        if abs(val)<0.01: return "sb","db",f"<b>Vega faible ({val:.4f} €/%)</b> — La volatilité du marché vous affecte peu."
        elif val>0:        return "sg","dg",f"<b>Vega positif ({val:.4f} €/%)</b> — Si la volatilité implicite monte de 1%, vous gagnez {val:.4f} €. Vous profitez de l'incertitude du marché."
        else:              return "sr","dr",f"<b>Vega négatif ({val:.4f} €/%)</b> — Si la volatilité implicite monte de 1%, vous perdez {abs(val):.4f} €. Vous profitez de la stabilité du marché."
    elif name=="rho":
        if abs(val)<0.003: return "sb","db",f"<b>Rho neutre ({val:+.4f} €/%)</b> — Les variations de taux d'intérêt ne vous affectent pas."
        elif val>0:        return "sg","dg",f"<b>Rho positif ({val:+.4f} €/%)</b> — Une hausse des taux vous est favorable (typique des calls)."
        else:              return "sr","dr",f"<b>Rho négatif ({val:+.4f} €/%)</b> — Une hausse des taux vous est défavorable (typique des puts)."
    return "sb","db",str(val)

def gamma_theta_msg(g,t):
    if g>0.003 and t<-0.001: return "so","do","<b>Long Gamma / Short Theta</b> — Vous achetez la convexité et payez le temps. Vous profitez des gros mouvements et perdez sur un marché calme."
    elif g<-0.003 and t>0.001: return "sg","dg","<b>Short Gamma / Long Theta</b> — Vous vendez la convexité et encaissez le temps. Vous profitez d'un marché calme et perdez sur les gros mouvements."
    return "sb","db",f"<b>Gamma/Theta équilibrés</b> — Gamma={g:.5f} · Theta={t:+.5f} €/j"

# ─────────────────────────────────────────────────────────────
#  DASHBOARD — 5 SVG charts
# ─────────────────────────────────────────────────────────────
def build_dashboard(S, K, T, r, sigma, q, otype):
    N = 200
    SR   = np.linspace(max(S*0.45,1), S*1.55, N)
    sigR = np.linspace(0.02, 1.0, N)
    TR   = np.linspace(0.005, max(T*2.5,0.5), N)

    prices = np.array([bs_price(s,K,T,r,sigma,q,otype) for s in SR])
    gs_all = [bs_greeks(s,K,T,r,sigma,q,otype) for s in SR]
    deltas = np.array([g["delta"] for g in gs_all])
    gammas = np.array([g["gamma"] for g in gs_all])
    intr   = np.array([max(s-K,0) if otype=="call" else max(K-s,0) for s in SR])
    p_sig  = np.array([bs_price(S,K,T,r,s,q,otype) for s in sigR])
    p_T    = np.array([bs_price(S,K,t,r,sigma,q,otype) for t in TR])
    cur    = bs_price(S,K,T,r,sigma,q,otype)
    G      = bs_greeks(S,K,T,r,sigma,q,otype)

    W, H = 680, 250

    def vl(xv, col, lbl="", dash=True):
        return {"x":xv,"color":col,"label":lbl,"dash":dash}

    # ① Prix
    svg1 = svg_chart([
        {"x":list(SR),"y":list(intr),"color":"#52525b","width":1,"dash":True,"label":"Valeur intrinsèque"},
        {"x":list(SR),"y":list(prices),"color":"#f59e0b","width":2.2,"fill":True,"label":f"Prix {otype.upper()}"},
    ], W=W, H=H, xlabel="Spot (€)", ylabel="Prix (€)",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b",f"K={K:.0f}")],
       show_dot={"x":S,"y":cur,"color":"#f59e0b","label":f"€{cur:.3f}"},
       title="Prix de l'option selon le spot")

    # ② Delta
    svg2 = svg_chart([
        {"x":list(SR),"y":list(deltas),"color":"#22c55e","width":2.2,"fill":True,"fill_color":"#22c55e","label":"Δ Delta"},
    ], W=W, H=H, xlabel="Spot (€)", ylabel="Delta",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b")],
       hline_zero=True,
       show_dot={"x":S,"y":G["delta"],"color":"#22c55e","label":f"{G['delta']:.4f}"},
       title="Delta — sensibilité au prix")

    # ③ Gamma
    svg3 = svg_chart([
        {"x":list(SR),"y":list(gammas),"color":"#a78bfa","width":2.2,"fill":True,"fill_color":"#a78bfa","label":"Γ Gamma"},
    ], W=W, H=H, xlabel="Spot (€)", ylabel="Gamma",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b")],
       hline_zero=True,
       show_dot={"x":S,"y":G["gamma"],"color":"#a78bfa","label":f"{G['gamma']:.5f}"},
       title="Gamma — convexité")

    # ④ Prix vs Vol
    svg4 = svg_chart([
        {"x":list(sigR*100),"y":list(p_sig),"color":"#ef4444","width":2.2,"fill":True,"fill_color":"#ef4444","label":"Prix"},
    ], W=W, H=H, xlabel="Volatilité implicite (%)", ylabel="Prix (€)",
       vlines=[vl(sigma*100,"#f59e0b",f"σ={sigma*100:.1f}%")],
       show_dot={"x":sigma*100,"y":cur,"color":"#f59e0b","label":f"€{cur:.3f}"},
       title="Sensibilité à la volatilité")

    # ⑤ Time Decay
    svg5 = svg_chart([
        {"x":list(TR),"y":list(p_T),"color":"#3b82f6","width":2.2,"fill":True,"fill_color":"#3b82f6","label":"Prix"},
    ], W=W, H=H, xlabel="Maturité (ans)", ylabel="Prix (€)",
       vlines=[vl(T,"#f59e0b",fmt_mat(T))],
       show_dot={"x":T,"y":cur,"color":"#f59e0b","label":f"€{cur:.3f}"},
       title="Érosion temporelle (Time Decay)")

    return svg1,svg2,svg3,svg4,svg5

# ─────────────────────────────────────────────────────────────
#  PAYOFF CHART
# ─────────────────────────────────────────────────────────────
def build_payoff(name, S, K, Tc, Tp, r, sc, sp, W=900, H=320):
    SR = np.linspace(S*0.50, S*1.50, 400)
    dk = S*0.07
    c  = lambda k,T,sg: bs_price(S,k,T,r,sg,0,"call")
    p  = lambda k,T,sg: bs_price(S,k,T,r,sg,0,"put")
    mx = np.maximum
    C0=c(K,Tc,sc); P0=p(K,Tp,sp)
    Kc,Kp=K+dk,K-dk
    K1,K3=K-dk,K+dk
    CK1=c(K1,Tc,sc); CK3=c(K3,Tc,sc)
    Ka,Kb,Kcc,Kd=K-2*dk,K-dk,K+dk,K+2*dk
    Pa=p(Ka,Tp,sp); Pb=p(Kb,Tp,sp); Pc=c(Kcc,Tc,sc); Pd=c(Kd,Tc,sc)

    pnls = {
        "Long Straddle":   mx(SR-K,0)+mx(K-SR,0)-C0-P0,
        "Short Straddle":  -(mx(SR-K,0)+mx(K-SR,0))+C0+P0,
        "Long Strangle":   mx(SR-Kc,0)+mx(Kp-SR,0)-c(Kc,Tc,sc)-p(Kp,Tp,sp),
        "Short Strangle":  -(mx(SR-Kc,0)+mx(Kp-SR,0))+c(Kc,Tc,sc)+p(Kp,Tp,sp),
        "Bull Call Spread": mx(SR-K,0)-mx(SR-K3,0)-(C0-CK3),
        "Bear Put Spread":  mx(K3-SR,0)-mx(K-SR,0)-(p(K3,Tp,sp)-P0),
        "Long Butterfly":   mx(SR-K1,0)-2*mx(SR-K,0)+mx(SR-K3,0)-(CK1-2*C0+CK3),
        "Short Butterfly":  -(mx(SR-K1,0)-2*mx(SR-K,0)+mx(SR-K3,0))+(CK1-2*C0+CK3),
        "Iron Condor":     mx(Kb-SR,0)-mx(Ka-SR,0)+mx(SR-Kcc,0)-mx(SR-Kd,0)+(-Pb+Pa-Pc+Pd),
        "Iron Butterfly":  -mx(SR-K,0)-mx(K-SR,0)+mx(SR-K3,0)+mx(K1-SR,0)+(-C0-P0+CK3+CK1),
        "Long Call":       mx(SR-K,0)-C0,
        "Long Put":        mx(K-SR,0)-P0,
        "Covered Call":    (SR-S)-mx(SR-K3,0)+c(K3,Tc,sc),
        "Protective Put":  (SR-S)+mx(K1-SR,0)-p(K1,Tp,sp),
    }
    pnl = pnls.get(name, np.zeros_like(SR))
    col = STRATEGIES[name]["color"]

    vlines = [{"x":S,"color":"#3b82f6","label":f"S={S:.0f}","dash":True},
               {"x":K,"color":"#52525b","label":f"K={K:.0f}","dash":True}]
    # break-evens as vlines
    idxs = np.where(np.diff(np.sign(pnl)))[0]
    for idx in idxs:
        be = (SR[idx]+SR[idx+1])/2
        vlines.append({"x":be,"color":"#f59e0b","label":f"BE {be:.1f}","dash":True})

    return svg_chart([
        {"x":list(SR),"y":list(pnl),"color":col,"width":2.5,"fill_pos_neg":True,"label":"P&L"},
    ], W=W, H=H, xlabel="Prix à l'expiration (€)", ylabel="P&L (€)",
       hline_zero=True, vlines=vlines,
       title=f"{name} — Profit / Perte à l'expiration",
       PAD_L=58, PAD_R=20, PAD_T=30, PAD_B=42)

# ─────────────────────────────────────────────────────────────
#  CUSTOM PAYOFF + GREEKS CHART
# ─────────────────────────────────────────────────────────────
PALETTE = ["#3b82f6","#22c55e","#f59e0b","#a78bfa","#ef4444","#06b6d4"]

def build_custom_payoff(legs, S_ref, name, W=920, H=340):
    SR = np.linspace(S_ref*0.5, S_ref*1.5, 400)
    total = np.zeros_like(SR); net_prem=0.0; series=[]
    for i,leg in enumerate(legs):
        if not leg["active"]: continue
        pr   = bs_price(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],0,leg["inst"])
        cost = leg["dir"]*pr*leg["qty"]; net_prem+=cost
        pnl  = (leg["dir"]*np.maximum(SR-leg["K"],0)*leg["qty"]-cost
                if leg["inst"]=="call"
                else leg["dir"]*np.maximum(leg["K"]-SR,0)*leg["qty"]-cost)
        total += pnl
        series.append({"x":list(SR),"y":list(pnl),"color":PALETTE[i%6],
                       "width":1.2,"dash":True,"label":leg["label"]})
    series.append({"x":list(SR),"y":list(total),"color":"#ffffff","width":3,
                   "fill_pos_neg":True,"label":"P&L total"})

    vlines=[{"x":S_ref,"color":"#3b82f6","label":f"S₀={S_ref:.0f}","dash":True}]
    idxs=np.where(np.diff(np.sign(total)))[0]
    for idx in idxs:
        be=(SR[idx]+SR[idx+1])/2
        vlines.append({"x":be,"color":"#f59e0b","label":f"BE {be:.1f}","dash":True})

    nc="encaissée" if net_prem<0 else "payée"
    svg = svg_chart(series, W=W, H=H,
                    xlabel="Prix à l'expiration (€)", ylabel="P&L (€)",
                    hline_zero=True, vlines=vlines,
                    title=f"{name}  ·  Prime nette {nc} : €{abs(net_prem):.4f}",
                    PAD_L=60, PAD_R=20, PAD_T=32, PAD_B=44)
    return svg, total, net_prem

def build_custom_greeks(legs, S_ref, W=920, H=200):
    SR=np.linspace(S_ref*0.5,S_ref*1.5,150)
    PD=np.zeros_like(SR); PG=np.zeros_like(SR)
    PT=np.zeros_like(SR); PV=np.zeros_like(SR)
    for leg in legs:
        if not leg["active"]: continue
        for i,s in enumerate(SR):
            G=bs_greeks(s,leg["K"],leg["T"],leg["r"],leg["sigma"],0,leg["inst"])
            PD[i]+=leg["dir"]*leg["qty"]*G["delta"]
            PG[i]+=leg["dir"]*leg["qty"]*G["gamma"]
            PT[i]+=leg["dir"]*leg["qty"]*G["theta"]
            PV[i]+=leg["dir"]*leg["qty"]*G["vega"]

    vl=[{"x":S_ref,"color":"#3b82f6","dash":True}]
    kwargs=dict(W=W//4, H=H, hline_zero=True, vlines=vl,
                xlabel="Spot (€)", PAD_L=48,PAD_R=12,PAD_T=26,PAD_B=36)

    svgs=[]
    for data,col,title in [(PD,"#22c55e","Delta"),(PG,"#a78bfa","Gamma"),
                            (PT,"#f59e0b","Theta"),(PV,"#3b82f6","Vega")]:
        s=svg_chart([{"x":list(SR),"y":list(data),"color":col,"width":2,
                      "fill":True,"fill_color":col}],
                    title=title,**kwargs)
        svgs.append(s)
    return svgs

# ─────────────────────────────────────────────────────────────
#  STRATEGIES
# ─────────────────────────────────────────────────────────────
STRATEGIES={
    "Long Straddle":   {"desc":"Acheter call + put au même strike. Profite d'un grand mouvement dans n'importe quel sens.",
     "legs":[("buy","call","K ATM"),("buy","put","K ATM")],"outlook":"Grand mouvement attendu",
     "max_gain":"Illimité","max_loss":"Prime totale","be":"K ± prime",
     "greeks":"Long Gamma · Short Theta · Long Vega · Delta ≈ 0","color":"#22c55e"},
    "Short Straddle":  {"desc":"Vendre call + put au même strike. Encaisse la prime si le marché reste stable.",
     "legs":[("sell","call","K ATM"),("sell","put","K ATM")],"outlook":"Marché calme attendu",
     "max_gain":"Prime totale","max_loss":"Illimité","be":"K ± prime",
     "greeks":"Short Gamma · Long Theta · Short Vega · Delta ≈ 0","color":"#ef4444"},
    "Long Strangle":   {"desc":"Acheter call OTM + put OTM. Moins cher que le straddle, mais le marché doit bouger encore plus.",
     "legs":[("buy","call","K+7%"),("buy","put","K−7%")],"outlook":"Très grand mouvement",
     "max_gain":"Illimité","max_loss":"Prime totale","be":"K± + prime",
     "greeks":"Long Gamma · Short Theta · Long Vega · Delta ≈ 0","color":"#3b82f6"},
    "Short Strangle":  {"desc":"Vendre call OTM + put OTM. Zone de gain plus large que le straddle court.",
     "legs":[("sell","call","K+7%"),("sell","put","K−7%")],"outlook":"Marché range",
     "max_gain":"Prime totale","max_loss":"Illimité","be":"K± + prime",
     "greeks":"Short Gamma · Long Theta · Short Vega","color":"#f59e0b"},
    "Bull Call Spread": {"desc":"Acheter un call, vendre un call plus élevé. Exposition haussière limitée à moindre coût.",
     "legs":[("buy","call","K1"),("sell","call","K2>K1")],"outlook":"Haussier modéré",
     "max_gain":"(K2−K1)−prime","max_loss":"Prime nette","be":"K1 + prime",
     "greeks":"Delta + · Gamma faible · Theta neutre · Vega faible","color":"#22c55e"},
    "Bear Put Spread":  {"desc":"Acheter un put, vendre un put inférieur. Exposition baissière limitée à moindre coût.",
     "legs":[("buy","put","K2"),("sell","put","K1<K2")],"outlook":"Baissier modéré",
     "max_gain":"(K2−K1)−prime","max_loss":"Prime nette","be":"K2 − prime",
     "greeks":"Delta − · Gamma faible · Theta neutre · Vega faible","color":"#a78bfa"},
    "Long Butterfly":  {"desc":"Acheter les ailes K1 et K3, vendre 2× le corps K2. Gagne si le prix expire exactement à K2.",
     "legs":[("buy","call","K1"),("sell","call","K2 ×2"),("buy","call","K3")],"outlook":"Marché très stable",
     "max_gain":"(K2−K1)−prime","max_loss":"Prime nette","be":"K1+prime · K3−prime",
     "greeks":"Delta ≈ 0 · Short Gamma · Long Theta · Vega faible","color":"#f59e0b"},
    "Short Butterfly": {"desc":"Inverse du butterfly. Vendre les ailes, acheter le corps. Gagne si le prix s'éloigne de K2.",
     "legs":[("sell","call","K1"),("buy","call","K2 ×2"),("sell","call","K3")],"outlook":"Mouvement attendu",
     "max_gain":"Prime nette","max_loss":"(K2−K1)−prime","be":"K1+prime · K3−prime",
     "greeks":"Long Gamma · Short Theta · Vega faible","color":"#ef4444"},
    "Iron Condor":     {"desc":"Vendre put spread + call spread. Gagne si le prix reste dans le couloir à l'expiration.",
     "legs":[("buy","put","K1"),("sell","put","K2"),("sell","call","K3"),("buy","call","K4")],"outlook":"Marché range · Risque limité",
     "max_gain":"Prime nette","max_loss":"Largeur − prime","be":"K2−prime · K3+prime",
     "greeks":"Delta ≈ 0 · Short Gamma · Long Theta · Short Vega","color":"#3b82f6"},
    "Iron Butterfly":  {"desc":"Vendre straddle ATM + acheter strangle OTM. Gain max si expiration au strike central.",
     "legs":[("buy","put","K1"),("sell","put","K ATM"),("sell","call","K ATM"),("buy","call","K3")],"outlook":"Très stable",
     "max_gain":"Prime nette","max_loss":"Largeur − prime","be":"K ± prime",
     "greeks":"Delta ≈ 0 · Short Gamma fort · Long Theta · Short Vega","color":"#f59e0b"},
    "Long Call":       {"desc":"Droit d'acheter à prix fixe. Gain illimité si le prix monte, perte limitée à la prime.",
     "legs":[("buy","call","K")],"outlook":"Haussier",
     "max_gain":"Illimité","max_loss":"Prime payée","be":"K + prime",
     "greeks":"Delta + · Gamma + · Theta − · Vega +","color":"#22c55e"},
    "Long Put":        {"desc":"Droit de vendre à prix fixe. Sert de protection ou d'exposition baissière.",
     "legs":[("buy","put","K")],"outlook":"Baissier ou protection",
     "max_gain":"K − prime","max_loss":"Prime payée","be":"K − prime",
     "greeks":"Delta − · Gamma + · Theta − · Vega +","color":"#a78bfa"},
    "Covered Call":    {"desc":"Détenir l'action et vendre un call. Génère un revenu, plafonne le gain à la hausse.",
     "legs":[("buy","stock","actions"),("sell","call","K>S")],"outlook":"Neutre · Revenu",
     "max_gain":"(K−S)+prime","max_loss":"S−prime","be":"S − prime",
     "greeks":"Delta réduit · Short Gamma · Long Theta · Short Vega","color":"#f59e0b"},
    "Protective Put":  {"desc":"Détenir l'action et acheter un put. Assurance contre une baisse, gain illimité à la hausse.",
     "legs":[("buy","stock","actions"),("buy","put","K")],"outlook":"Haussier avec protection",
     "max_gain":"Illimité","max_loss":"S−K+prime","be":"S + prime",
     "greeks":"Delta + · Gamma + · Theta − · Vega +","color":"#3b82f6"},
}
LEG_A={"buy":("tby","Achat"),"sell":("tse","Vente")}
LEG_I={"call":("tca","Call"),"put":("tpu","Put"),"stock":("tca","Action")}

def legs_html(legs):
    h=""
    for a,ins,strike in legs:
        ac,at=LEG_A.get(a,("tby",a)); ic,it=LEG_I.get(ins,("tca",ins))
        h+=f'<span class="tag {ac}">{at}</span><span class="tag {ic}">{it}</span>'
        h+=f'<span style="font-size:.58rem;color:#3f3f46;margin:0 3px">{strike}</span>'
    return h

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-l">
    <div class="hdr-ico">◈</div>
    <div><div class="hdr-t">Options Lab</div><div class="hdr-s">Pricer Black-Scholes · Greeks · Stratégies</div></div>
  </div>
  <div class="live">● Live</div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SIDEBAR — Paramètres Pricer (partagés avec tous les onglets)
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">◈ Options Lab — Paramètres</div>', unsafe_allow_html=True)

    # ── Ticker marché réel ──────────────────────────────────
    try:
        import yfinance as _yf
        _YF_OK = True
    except ImportError:
        _YF_OK = False

    if _YF_OK:
        st.markdown('<div class="sb-title">📡 Données marché</div>', unsafe_allow_html=True)
        _ticker = st.text_input("Ticker Yahoo Finance",
                                value=st.session_state.get("last_ticker", "AAPL"),
                                placeholder="AAPL · MC.PA · ^FCHI",
                                key="sb_ticker",
                                help="Exemples : AAPL, TSLA, MC.PA, SAN.PA, ^FCHI")
        if st.button("⬇ Charger spot & vol", key="sb_load_mkt", use_container_width=True):
            try:
                with st.spinner("Yahoo Finance…"):
                    tk   = _yf.Ticker(_ticker.upper().strip())
                    info = tk.fast_info
                    spot = float(info.get("last_price") or info.get("regularMarketPrice") or 0)
                    if spot <= 0:
                        raise ValueError(f"Ticker '{_ticker}' introuvable.")
                    hist = tk.history(period="1y")
                    if len(hist) > 5:
                        log_ret    = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                        hist_sigma = float(log_ret.std() * np.sqrt(252))
                    else:
                        hist_sigma = 0.20
                    st.session_state.shared_S     = round(spot, 2)
                    st.session_state.shared_K     = round(spot, 2)
                    st.session_state.shared_sigma = round(hist_sigma * 100, 1)
                    st.session_state.last_ticker  = _ticker
                    st.success(f"✅ {_ticker.upper()} — {spot:.2f} · vol {hist_sigma*100:.1f}%")
                    st.rerun()
            except Exception as _e:
                st.error(str(_e))
    else:
        st.caption("📡 `pip install yfinance` pour les données réelles")

    st.markdown("---")

    # ── Type d'option ───────────────────────────────────────
    st.markdown('<div class="sb-title">Type d\'option</div>', unsafe_allow_html=True)
    otype = st.radio("", ["call", "put"], horizontal=True, key="ot1",
                     help="Call : droit d'acheter · Put : droit de vendre")
    _badge_cls = "sb-call" if otype == "call" else "sb-put"
    _badge_lbl = "● CALL — droit d'acheter" if otype == "call" else "● PUT — droit de vendre"
    st.markdown(f'<div class="sb-badge {_badge_cls}">{_badge_lbl}</div>', unsafe_allow_html=True)

    # ── Sous-jacent & Strike ────────────────────────────────
    st.markdown('<div class="sb-title">Sous-jacent & Strike</div>', unsafe_allow_html=True)
    fl("Spot S₀  (€)")
    S = st.number_input("S0", value=st.session_state.get("shared_S", 100.0),
                        step=1.0, label_visibility="collapsed",
                        help="Prix actuel de l'actif sur le marché", key="t1_S")
    st.session_state["shared_S"] = S

    fl("Strike K  (€)")
    K = st.number_input("K", value=st.session_state.get("shared_K", 100.0),
                        step=1.0, label_visibility="collapsed",
                        help="Prix d'exercice de l'option", key="t1_K")
    st.session_state["shared_K"] = K

    mr = S / K if K > 0 else 1
    if abs(mr - 1) < 0.015:                                          pc, pt = "matm", "ATM"
    elif (otype == "call" and S > K) or (otype == "put" and S < K):  pc, pt = "mitm", "ITM"
    else:                                                             pc, pt = "motm", "OTM"
    st.markdown(f'<span class="mpill {pc}">{pt} · {mr:.3f}</span>', unsafe_allow_html=True)

    # ── Maturité ────────────────────────────────────────────
    st.markdown('<div class="sb-title">Maturité</div>', unsafe_allow_html=True)
    _mc1, _mc2, _mc3 = st.columns(3)
    with _mc1: _ty = st.number_input("A", 0, 30, 1, 1, key="p1_y",
                                      label_visibility="collapsed", help="Années")
    with _mc2: _tm = st.number_input("M", 0, 11, 0, 1, key="p1_m",
                                      label_visibility="collapsed", help="Mois")
    with _mc3: _td = st.number_input("J", 0, 30, 0, 1, key="p1_d",
                                      label_visibility="collapsed", help="Jours")
    T = mat_from_ymd(_ty, _tm, _td)
    st.markdown(f'<span class="mb">{fmt_mat(T)} · {T:.3f}a</span>', unsafe_allow_html=True)

    # ── Taux, Dividende, Volatilité ─────────────────────────
    st.markdown('<div class="sb-title">Paramètres de marché</div>', unsafe_allow_html=True)
    fl("Taux sans risque  r  (%)")
    r = st.slider("r", 0.0, 15.0,
                  float(st.session_state.get("shared_r", 5.0)), 0.1,
                  label_visibility="collapsed",
                  help="Taux d'intérêt annuel (BCE ≈ 3–4% en 2024)", key="t1_r") / 100
    st.session_state["shared_r"] = r * 100

    fl("Dividende  q  (%)")
    q_div = st.slider("q", 0.0, 10.0,
                      float(st.session_state.get("shared_q", 0.0)), 0.1,
                      label_visibility="collapsed",
                      help="Dividende annuel continu. 0 si pas de dividende.", key="t1_q") / 100
    st.session_state["shared_q"] = q_div * 100

    fl("Volatilité  σ  (%)")
    sigma = st.slider("sg", 1.0, 150.0,
                      float(st.session_state.get("shared_sigma", 20.0)), 0.5,
                      label_visibility="collapsed",
                      help="Volatilité implicite annualisée.", key="t1_sigma") / 100
    st.session_state["shared_sigma"] = sigma * 100

    # ── Volatilité implicite ────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sb-title">Volatilité implicite</div>', unsafe_allow_html=True)
    fl("Prix de marché observé (€)")
    mkt = st.number_input("Prix observé (€)", value=0.0, step=0.01, min_value=0.0,
                          label_visibility="collapsed",
                          help="Prix affiché chez votre courtier → retrouve la vol. implicite",
                          key="sb_mkt")
    if mkt > 0:
        iv = implied_vol(mkt, S, K, T, r, q_div, otype)
        if not np.isnan(iv):
            st.markdown(f'<div class="ivr">σ impl. = {iv*100:.3f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ivf">Prix hors bornes BS</div>', unsafe_allow_html=True)

    # ── Sync autres onglets ─────────────────────────────────
    st.markdown("---")
    if st.button("⇄ Sync Stratégies & Builder", use_container_width=True,
                 help="Propage Spot, Vol et Taux vers les onglets Stratégies et Strategy Builder"):
        st.rerun()

tab1, tab2, tab3 = st.tabs(["Pricer & Greeks", "Stratégies", "Strategy Builder"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — Zone principale pleine largeur
# ═══════════════════════════════════════════════════════════
with tab1:
    price=bs_price(S,K,T,r,sigma,q_div,otype)
    G=bs_greeks(S,K,T,r,sigma,q_div,otype)
    d1v=(np.log(S/K)+(r-q_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T)) if T>1e-9 and sigma>1e-9 else 0
    d2v=d1v-sigma*np.sqrt(T) if T>1e-9 else 0
    bc="ph-c" if otype=="call" else "ph-p"
    bt="CALL" if otype=="call" else "PUT"

    # ── Prix hero + Greeks côte à côte ──────────────────────
    _hero_c, _grk_c = st.columns([5, 7], gap="large")
    with _hero_c:
        st.markdown(f"""
        <div class="ph">
          <div>
            <div class="ph-ey">Prix Black-Scholes</div>
            <div class="ph-row"><span class="ph-cur">€</span><span class="ph-val">{price:.4f}</span></div>
            <div class="ph-sub">
              <span>d₁ = {d1v:.4f}</span><span>d₂ = {d2v:.4f}</span>
              <span>N(d₁) = {norm.cdf(d1v):.4f}</span><span>N(d₂) = {norm.cdf(d2v):.4f}</span>
            </div>
          </div>
          <span class="ph-badge {bc}">{bt}</span>
        </div>""", unsafe_allow_html=True)

        sh("Exposition — lecture de vos risques")
        ic1,ic2=st.columns(2)
        with ic1:
            for gn in ["delta","gamma","theta"]: a,b,c_=interp(gn,G[gn]); sig(a,b,c_)
        with ic2:
            for gn in ["vega","rho"]: a,b,c_=interp(gn,G[gn]); sig(a,b,c_)
            a,b,c_=gamma_theta_msg(G["gamma"],G["theta"]); sig(a,b,c_)

    with _grk_c:
        sh("Grecs")
        cols=st.columns(6)
        gdata=[("Δ","Delta",G["delta"],".4f","#22c55e","Variation si le sous-jacent bouge de 1 €"),
               ("Γ","Gamma",G["gamma"],".5f","#a78bfa","Vitesse de changement du delta"),
               ("Θ","Theta",G["theta"],"+.4f","#f59e0b","€ gagnés ou perdus chaque jour"),
               ("ν","Vega", G["vega"], ".4f","#3b82f6","€ gagnés si la volatilité monte de 1%"),
               ("ρ","Rho",  G["rho"],  "+.4f","#ef4444","€ gagnés si les taux montent de 1%"),
               ("Λ","Vanna",G["vanna"],"+.4f","#71717a","Sensibilité croisée prix × volatilité")]
        for col,(sym,nm,v,fmt,col_c,desc) in zip(cols,gdata):
            with col: greek_card(sym,nm,v,fmt,col_c,desc)

        with st.expander("Tableau détaillé des grecs"):
            df_g=pd.DataFrame([
                {"Grec":"Δ Delta","Valeur":f"{G['delta']:+.6f}","Unité":"€/€","Sens":"+ si haussier"},
                {"Grec":"Γ Gamma","Valeur":f"{G['gamma']:.6f}","Unité":"—","Sens":"+ si convexité favorable"},
                {"Grec":"Θ Theta","Valeur":f"{G['theta']:+.6f}","Unité":"€/j","Sens":"+ si le temps aide"},
                {"Grec":"ν Vega","Valeur":f"{G['vega']:.6f}","Unité":"€/%","Sens":"+ si vol. implicite ↑"},
                {"Grec":"ρ Rho","Valeur":f"{G['rho']:+.6f}","Unité":"€/%","Sens":"+ si taux ↑"},
                {"Grec":"Λ Vanna","Valeur":f"{G['vanna']:+.6f}","Unité":"—","Sens":"Cross delta/vol"},
            ])
            st.dataframe(df_g.set_index("Grec"),use_container_width=True)

    # ── Visualisations pleine largeur ────────────────────────
    sh("Visualisations")
    svg1,svg2,svg3,svg4,svg5 = build_dashboard(S,K,T,r,sigma,q_div,otype)

    # Ligne 1 : 3 colonnes égales
    r1c1,r1c2,r1c3 = st.columns(3)
    with r1c1: show_svg(svg1)
    with r1c2: show_svg(svg2)
    with r1c3: show_svg(svg3)

    # Ligne 2 : 2 colonnes larges
    r2c1,r2c2 = st.columns(2)
    with r2c1: show_svg(svg4)
    with r2c2: show_svg(svg5)

# ═══════════════════════════════════════════════════════════
#  TAB 2
# ═══════════════════════════════════════════════════════════
with tab2:
    p1,p2,p3=st.columns(3)
    with p1:
        S2=st.number_input("Spot S₀",value=100.0,step=1.0,key="s2",help="Prix actuel du sous-jacent")
        K2=st.number_input("Strike K central",value=100.0,step=1.0,key="k2",
                           help="Strike ATM. Les strikes OTM sont calculés à ±7% automatiquement")
        r2=st.number_input("Taux r (%)",value=5.0,step=0.1,key="r2",help="Taux sans risque (%)") / 100
    with p2:
        fl("Maturité Calls  (A / M / J)"); Tc2=mat_inline("s2c",1,0,0)
        sig_c2=st.slider("Vol. Calls σ (%)",1.0,100.0,20.0,0.5,key="sc2",
                         help="Volatilité implicite utilisée pour pricer les calls") / 100
    with p3:
        fl("Maturité Puts  (A / M / J)"); Tp2=mat_inline("s2p",1,0,0)
        sig_p2=st.slider("Vol. Puts σ (%)",1.0,100.0,20.0,0.5,key="sp2",
                         help="Volatilité implicite utilisée pour pricer les puts") / 100

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sh("Choisir une stratégie")

    if "sel" not in st.session_state: st.session_state.sel="Long Straddle"
    snames=list(STRATEGIES.keys())
    for row in [snames[i:i+3] for i in range(0,len(snames),3)]:
        cols=st.columns(3)
        for col,name in zip(cols,row):
            with col:
                info=STRATEGIES[name]; sel=st.session_state.sel==name
                st.markdown(f'<div class="sc {"sc-sel" if sel else ""}"><div class="sc-nm">{name}</div>'
                            f'<div style="margin:5px 0 4px">{legs_html(info["legs"])}</div>'
                            f'<div class="sc-s">{info["outlook"]}</div></div>', unsafe_allow_html=True)
                if st.button("Sélectionner",key=f"s2_{name}"):
                    st.session_state.sel=name; st.rerun()

    st.markdown("---")
    strat=st.session_state.sel; info=STRATEGIES[strat]

    d1c,d2c,d3c=st.columns([2,1,1])
    with d1c:
        st.markdown(f'<div class="card" style="border-left:3px solid {info["color"]}">'
                    f'<div style="font-size:.96rem;font-weight:700;color:{info["color"]};margin-bottom:7px">{strat}</div>'
                    f'<div style="font-size:.76rem;color:#a1a1aa;line-height:1.7;margin-bottom:10px">{info["desc"]}</div>'
                    f'{legs_html(info["legs"])}</div>', unsafe_allow_html=True)
    with d2c:
        st.markdown(f'<div class="card" style="height:100%"><div class="ct">Payoff</div>'
                    f'<div style="font-size:.73rem;line-height:2.1;color:#d4d4d8">'
                    f'📈 <b>Gain max</b><br>{info["max_gain"]}<br>'
                    f'📉 <b>Perte max</b><br>{info["max_loss"]}<br>'
                    f'⚖️ <b>Seuil rentabilité</b><br>{info["be"]}</div></div>', unsafe_allow_html=True)
    with d3c:
        tips=[]
        if "Long Gamma" in info["greeks"]:  tips.append("⚡ <b>Long Gamma</b> — profite des grands mouvements, perd le temps")
        if "Short Gamma" in info["greeks"]: tips.append("💰 <b>Short Gamma</b> — encaisse le temps, perd sur les grands mouvements")
        if "Long Vega" in info["greeks"]:   tips.append("📈 <b>Long Vega</b> — profite si la volatilité monte")
        if "Short Vega" in info["greeks"]:  tips.append("📉 <b>Short Vega</b> — profite si la volatilité reste basse")
        st.markdown(f'<div class="card" style="height:100%"><div class="ct">Greeks</div>'
                    f'<div style="font-size:.72rem;color:#a1a1aa;margin-bottom:8px">{info["greeks"]}</div>'
                    f'<div style="font-size:.70rem;color:#52525b;line-height:2.0">{"<br>".join(tips)}</div></div>',
                    unsafe_allow_html=True)

    sh("Profil de gain/perte à l'expiration")
    st.markdown(f'<div style="font-size:.68rem;color:#52525b;margin-bottom:6px">'
                f'Calls T={fmt_mat(Tc2)} σ={sig_c2*100:.1f}%  ·  Puts T={fmt_mat(Tp2)} σ={sig_p2*100:.1f}%</div>',
                unsafe_allow_html=True)
    show_svg(build_payoff(strat,S2,K2,Tc2,Tp2,r2,sig_c2,sig_p2), height=330)

    sh("Greeks indicatifs — leg ATM")
    G2c=bs_greeks(S2,K2,Tc2,r2,sig_c2,0,"call"); G2p=bs_greeks(S2,K2,Tp2,r2,sig_p2,0,"put")
    adj=-1 if strat in ["Short Straddle","Short Strangle","Short Butterfly"] else 1
    Gm={k:(G2c[k]+G2p[k])*adj/2 for k in G2c}
    gcols=st.columns(4)
    for col,(sym,nm,key,fmt,color) in zip(gcols,[("Δ","Delta","delta",".4f","#22c55e"),
        ("Γ","Gamma","gamma",".5f","#a78bfa"),("Θ","Theta","theta","+.5f","#f59e0b"),("ν","Vega","vega",".4f","#3b82f6")]):
        with col: greek_card(sym,nm,Gm[key],fmt,color,"")

    gi1,gi2=st.columns(2)
    with gi1:
        for gn in ["delta","gamma"]: a,b,c_=interp(gn,Gm[gn]); sig(a,b,c_)
    with gi2:
        for gn in ["theta","vega"]: a,b,c_=interp(gn,Gm[gn]); sig(a,b,c_)
        a,b,c_=gamma_theta_msg(Gm["gamma"],Gm["theta"]); sig(a,b,c_)

    with st.expander("Tableau comparatif — toutes les stratégies"):
        rows=[]
        for nm,inf in STRATEGIES.items():
            a=-1 if nm in ["Short Straddle","Short Strangle","Short Butterfly"] else 1
            Gi=bs_greeks(S2,K2,Tc2,r2,sig_c2,0,"call")
            rows.append({"Stratégie":nm,"Delta":f"{Gi['delta']*a:+.4f}","Gamma":f"{Gi['gamma']*a:+.5f}",
                "Theta":f"{Gi['theta']*a:+.5f}","Vega":f"{Gi['vega']*a:.4f}",
                "Vue":inf["outlook"],"Gain max":inf["max_gain"],"Perte max":inf["max_loss"]})
        st.dataframe(pd.DataFrame(rows).set_index("Stratégie"),use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card" style="margin-bottom:14px;font-size:.77rem;color:#a1a1aa;line-height:1.7">'
                'Construisez librement une stratégie en combinant jusqu\'à <b>6 jambes</b> indépendantes.<br>'
                'Chaque jambe a ses propres paramètres. Le graphique final agrège tous les P&L.</div>',
                unsafe_allow_html=True)

    hc1,hc2=st.columns([3,1])
    with hc1: sname=st.text_input("Nom de la stratégie",value="Ma stratégie",help="Affiché sur le graphique")
    with hc2: S_ref=st.number_input("Spot référence (€)",value=100.0,step=1.0,help="Centre de l'axe X du graphique")
    n_legs=st.slider("Nombre de jambes",1,6,2,1,help="Chaque jambe est une option indépendante")
    st.markdown("---")

    DFLTS=[
        dict(active=True, dir=1, inst="call",S=100.,K=100.,r=.05,sigma=.20,qty=1,y=1,m=0,d=0),
        dict(active=True, dir=-1,inst="call",S=100.,K=105.,r=.05,sigma=.20,qty=1,y=1,m=0,d=0),
        dict(active=False,dir=1, inst="put", S=100.,K=95., r=.05,sigma=.22,qty=1,y=1,m=0,d=0),
        dict(active=False,dir=-1,inst="put", S=100.,K=90., r=.05,sigma=.22,qty=1,y=1,m=0,d=0),
        dict(active=False,dir=1, inst="call",S=100.,K=110.,r=.05,sigma=.20,qty=1,y=1,m=0,d=0),
        dict(active=False,dir=-1,inst="put", S=100.,K=85., r=.05,sigma=.24,qty=1,y=1,m=0,d=0),
    ]
    LNAMES=["Jambe A","Jambe B","Jambe C","Jambe D","Jambe E","Jambe F"]

    legs_cfg=[]
    for i in range(n_legs):
        d=DFLTS[i]; lc2=PALETTE[i]; ln=LNAMES[i]
        h1,h2=st.columns([4,1])
        with h1: active=st.checkbox(f"**{ln}**",value=d["active"],key=f"la_{i}")
        with h2:
            direction=d["dir"]
            if active:
                dl=st.radio("",["Achat","Vente"],horizontal=True,
                            index=0 if d["dir"]==1 else 1,key=f"ldir_{i}",
                            help="Achat = vous payez la prime · Vente = vous encaissez la prime")
                direction=1 if dl=="Achat" else -1

        if active:
            dc="#22c55e" if direction==1 else "#ef4444"
            dt="● ACHAT" if direction==1 else "● VENTE"
            cls="lb" if direction==1 else "ls"
            st.markdown(f'<div class="lc {cls}" style="border-color:{lc2}">'
                        f'<div class="lh"><span class="ln" style="color:{lc2}">{ln}</span>'
                        f'<span style="font-size:.70rem;font-weight:600;color:{dc}">{dt}</span></div>',
                        unsafe_allow_html=True)
            lc1_,lc2_,lc3_,lc4_=st.columns(4)
            with lc1_:
                instrument=st.selectbox("Instrument",["call","put"],
                    index=0 if d["inst"]=="call" else 1,key=f"li_{i}",
                    help="Call : option d'achat · Put : option de vente")
                qty=st.number_input("Quantité",1,100,d["qty"],1,key=f"lq_{i}",help="Nombre de contrats")
            with lc2_:
                S_l=st.number_input("Spot S₀ (€)",value=d["S"],step=1.0,key=f"ls_{i}",
                    help="Prix actuel du sous-jacent pour cette jambe")
                K_l=st.number_input("Strike K (€)",value=d["K"],step=0.5,key=f"lk_{i}",
                    help="Prix d'exercice de cette option")
                mr2=S_l/K_l if K_l>0 else 1
                if abs(mr2-1)<0.02: mc2,mt2="matm","ATM"
                elif (instrument=="call" and mr2>1) or (instrument=="put" and mr2<1): mc2,mt2="mitm","ITM"
                else: mc2,mt2="motm","OTM"
                st.markdown(f'<span class="mpill {mc2}">{mt2} ({mr2:.3f})</span>', unsafe_allow_html=True)
            with lc3_:
                r_l=st.number_input("Taux r (%)",value=d["r"]*100,step=0.1,key=f"lr_{i}",
                    help="Taux sans risque (%)") / 100
                sig_l=st.slider(f"Vol. σ (%)",1.0,150.0,d["sigma"]*100,0.5,key=f"lsig_{i}",
                    help="Volatilité implicite annualisée") / 100
            with lc4_:
                fl("Maturité  (A / M / J)")
                y_l=st.number_input("A",0,30,d["y"],1,key=f"ly_{i}",label_visibility="collapsed",help="Années")
                m_l=st.number_input("M",0,11,d["m"],1,key=f"lm_{i}",label_visibility="collapsed",help="Mois")
                dj_l=st.number_input("J",0,30,d["d"],1,key=f"ld_{i}",label_visibility="collapsed",help="Jours")
                T_l=mat_from_ymd(y_l,m_l,dj_l)
                st.markdown(f'<span class="mb" style="font-size:.67rem">{fmt_mat(T_l)} · {T_l:.3f}a</span>',
                            unsafe_allow_html=True)
                prem=bs_price(S_l,K_l,T_l,r_l,sig_l,0,instrument)
                cost=direction*prem*qty
                cc="#ef4444" if direction==1 else "#22c55e"
                st.markdown(f'<div style="font-size:.69rem;color:#52525b">Unité : €{prem:.4f} × {qty}</div>'
                            f'<div style="font-size:.77rem;font-weight:700;color:{cc}">'
                            f'{"−" if direction==1 else "+"}€{abs(cost):.4f}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            legs_cfg.append(dict(active=True,dir=direction,inst=instrument,
                S=S_l,K=K_l,T=T_l,r=r_l,sigma=sig_l,qty=qty,
                label=f'{"Achat" if direction==1 else "Vente"} {instrument.upper()} K={K_l:.0f} ×{qty}'))
        else:
            st.markdown(f'<div class="lc lo"><span class="ln" style="color:#3f3f46">{ln} — désactivée</span></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    active_legs=[l for l in legs_cfg if l["active"]]

    if not active_legs:
        st.info("Activez au moins une jambe pour afficher le graphique.")
    else:
        net=sum(l["dir"]*bs_price(l["S"],l["K"],l["T"],l["r"],l["sigma"],0,l["inst"])*l["qty"] for l in active_legs)
        def pg(k): return sum(l["dir"]*l["qty"]*bs_greeks(l["S"],l["K"],l["T"],l["r"],l["sigma"],0,l["inst"])[k] for l in active_legs)
        PD,PG,PT,PV=pg("delta"),pg("gamma"),pg("theta"),pg("vega")
        nc="encaissée" if net<0 else "payée"; nc_col="#22c55e" if net<0 else "#ef4444"
        dc_col="#22c55e" if PD>0.05 else ("#ef4444" if PD<-0.05 else "#3b82f6")
        gc2="#22c55e" if PG>0 else "#ef4444"; tc2="#22c55e" if PT>0 else "#ef4444"; vc2="#22c55e" if PV>0 else "#ef4444"

        st.markdown(f"""
        <div class="tbar">
          <div class="tbi"><div class="tbl">Prime nette</div>
            <div class="tbv" style="color:{nc_col}">€{abs(net):.4f}</div>
            <div style="font-size:.58rem;color:#52525b">{nc}</div></div>
          <div class="tbi"><div class="tbl">Δ Delta</div><div class="tbv" style="color:{dc_col}">{PD:+.4f}</div></div>
          <div class="tbi"><div class="tbl">Γ Gamma</div><div class="tbv" style="color:{gc2}">{PG:+.5f}</div></div>
          <div class="tbi"><div class="tbl">Θ Theta €/j</div><div class="tbv" style="color:{tc2}">{PT:+.5f}</div></div>
          <div class="tbi"><div class="tbl">ν Vega €/%</div><div class="tbv" style="color:{vc2}">{PV:+.4f}</div></div>
        </div>""", unsafe_allow_html=True)

        sh("Exposition du portefeuille")
        bi1,bi2=st.columns(2)
        with bi1:
            for gn,gv in [("delta",PD),("gamma",PG)]: a,b,c_=interp(gn,gv); sig(a,b,c_)
        with bi2:
            for gn,gv in [("theta",PT),("vega",PV)]: a,b,c_=interp(gn,gv); sig(a,b,c_)
            a,b,c_=gamma_theta_msg(PG,PT); sig(a,b,c_)

        sh("Graphique P&L à l'expiration")
        svg_c,total_pnl,_=build_custom_payoff(active_legs,S_ref,sname)
        show_svg(svg_c, height=355)

        sh("Profils de Greeks vs Prix")
        greek_svgs=build_custom_greeks(active_legs,S_ref)
        gc1_,gc2_,gc3_,gc4_=st.columns(4)
        for col,svg in zip([gc1_,gc2_,gc3_,gc4_],greek_svgs):
            with col: show_svg(svg, height=215)

        mx_v=np.max(total_pnl); mn_v=np.min(total_pnl)
        be_n=len(np.where(np.diff(np.sign(total_pnl)))[0])
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
          <div class="tbi"><div class="tbl">Gain max borné</div><div class="tbv" style="color:#22c55e">€{mx_v:.2f}</div></div>
          <div class="tbi"><div class="tbl">Perte max bornée</div><div class="tbv" style="color:#ef4444">€{mn_v:.2f}</div></div>
          <div class="tbi"><div class="tbl">Break-evens</div><div class="tbv" style="color:#f59e0b">{be_n}</div></div>
          <div class="tbi"><div class="tbl">Jambes actives</div><div class="tbv" style="color:#3b82f6">{len(active_legs)}</div></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("Détail par jambe"):
            rows_l=[]
            for i,leg in enumerate(active_legs):
                pr=bs_price(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],0,leg["inst"])
                Gl=bs_greeks(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],0,leg["inst"])
                rows_l.append({"Jambe":LNAMES[i],"Sens":"Achat" if leg["dir"]==1 else "Vente",
                    "Type":leg["inst"].upper(),"S₀":f"{leg['S']:.1f}","Strike":f"{leg['K']:.1f}",
                    "Maturité":fmt_mat(leg["T"]),"σ%":f"{leg['sigma']*100:.1f}","Qté":leg["qty"],
                    "Prix":f"€{pr:.4f}","Coût net":f'{"−" if leg["dir"]==1 else "+"}€{pr*leg["qty"]:.4f}',
                    "Δ":f"{leg['dir']*Gl['delta']*leg['qty']:+.4f}",
                    "Γ":f"{leg['dir']*Gl['gamma']*leg['qty']:+.5f}",
                    "Θ":f"{leg['dir']*Gl['theta']*leg['qty']:+.5f}",
                    "ν":f"{leg['dir']*Gl['vega']*leg['qty']:+.4f}"})
            st.dataframe(pd.DataFrame(rows_l).set_index("Jambe"),use_container_width=True)

st.markdown('<div style="text-align:center;padding:32px 0 8px;font-size:.58rem;color:#1c1c1e;letter-spacing:1px">OPTIONS LAB · BLACK-SCHOLES · USAGE PÉDAGOGIQUE</div>', unsafe_allow_html=True)
