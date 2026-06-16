"""
Options Lab — Black-Scholes Pricer
----------------------------------
Auteur     : FloKov
Usage      : streamlit run app3.py
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

def bs_price_vec(S_arr, K, T, r, sigma, q=0.0, otype="call"):
    """Vectorized BS price for arrays."""
    S_arr = np.asarray(S_arr, dtype=float)
    if T <= 1e-9 or sigma <= 1e-9:
        if otype == "call": return np.maximum(S_arr*np.exp(-q*T) - K*np.exp(-r*T), 0)
        return np.maximum(K*np.exp(-r*T) - S_arr*np.exp(-q*T), 0)
    d1 = (np.log(S_arr/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if otype == "call": return S_arr*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S_arr*np.exp(-q*T)*norm.cdf(-d1)

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
    if mkt <= 0 or S <= 0 or K <= 0 or T <= 1e-9:
        return np.nan
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
TC_SVG  = "#d4d4d8"
T2_SVG  = "#e4e4e7"

def svg_chart(
    series_list,           # [{"x":[], "y":[], "color":"#hex", "label":"", "width":2, "dash":False, "fill":False, "fill_pos_neg":False}]
    W=700, H=260,
    title="",
    xlabel="", ylabel="",
    vlines=None,           # [{"x": val, "color":"#hex", "label":"", "dash":True}]
    hline_zero=False,
    show_dot=None,         # {"x":val,"y":val,"color":"#hex"}
    PAD_L=54, PAD_R=18, PAD_T=28, PAD_B=40,
    responsive=False,
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

    if responsive:
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
               f'preserveAspectRatio="xMidYMid meet" '
               f'style="background:{BG_SVG};border-radius:10px;display:block;width:100%;height:auto">']
    else:
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
                   f'font-family="DM Mono,monospace" font-size="10.5" fill="{TC_SVG}">{label_v}</text>')

    # Grid lines x
    n_xticks = 5
    for i in range(n_xticks+1):
        xv = xmin + i*(xmax-xmin)/n_xticks
        xp = tx(xv)
        svg.append(f'<line x1="{xp:.1f}" y1="{PAD_T}" x2="{xp:.1f}" y2="{PAD_T+ph}" '
                   f'stroke="{GR_SVG}" stroke-width="1" opacity=".5"/>')
        lv = f"{xv:.0f}" if abs(xv)>=1 else f"{xv:.2f}"
        svg.append(f'<text x="{xp:.1f}" y="{PAD_T+ph+14}" text-anchor="middle" '
                   f'font-family="DM Mono,monospace" font-size="10.5" fill="{TC_SVG}">{lv}</text>')

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
                lbl_y = PAD_T + ph + (PAD_B * 0.55)
                svg.append(f'<text x="{xp+3:.1f}" y="{lbl_y:.1f}" font-family="DM Mono,monospace" '
                           f'font-size="10" fill="{vl["color"]}" opacity=".9">{vl["label"]}</text>')

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
                       f'font-size="10.5" fill="{col}">{label}</text>')

    # Legend — multi-line wrap
    lx, ly = PAD_L+4, PAD_T+6
    max_lx = W - PAD_R - 10
    for s in series_list:
        if not s.get("label"): continue
        item_w = len(s["label"])*5.5 + 30
        if lx + item_w > max_lx:
            lx = PAD_L+4
            ly += 14
        svg.append(f'<rect x="{lx}" y="{ly-6}" width="14" height="3" rx="1.5" fill="{s["color"]}"/>')
        svg.append(f'<text x="{lx+18}" y="{ly+1}" font-family="Inter,sans-serif" '
                   f'font-size="10" fill="{T2_SVG}">{html.escape(s["label"])}</text>')
        lx += item_w

    # ── Hover bands (crosshair + tooltip) ───────────────────
    n_bands = 30
    if len(all_x) > 1 and pw > 10:
        band_w = pw / n_bands
        svg.append('<style>.hb .hl,.hb .ht,.hb .hbg{opacity:0;transition:opacity .1s ease}'
                   '.hb:hover .hl,.hb:hover .ht,.hb:hover .hbg{opacity:1}</style>')
        for bi in range(n_bands):
            bx = PAD_L + bi * band_w
            cx = bx + band_w / 2
            xv = xmin + (cx - PAD_L) / pw * (xmax - xmin)
            tip_lines = []
            for s in series_list:
                if not s.get("label") or len(s["x"]) < 2: continue
                yv = float(np.interp(xv, s["x"], s["y"]))
                vstr = f"{yv:.4f}" if abs(yv) < 10 else f"{yv:.2f}"
                tip_lines.append((s["label"], vstr, s.get("color", "#fff")))
            right_side = cx > PAD_L + pw * 0.6
            tx_off = -10 if right_side else 10
            t_anchor = "end" if right_side else "start"
            xv_str = f"{xv:.1f}" if abs(xv) >= 1 else f"{xv:.3f}"
            max_chars = max((len(l)+len(v)+3 for l,v,_ in tip_lines), default=10)
            bg_w = max(max_chars * 5.8 + 16, 80)
            bg_h = len(tip_lines) * 14 + 22
            bg_x = cx + tx_off - bg_w - 2 if right_side else cx + tx_off - 6
            bg_y = PAD_T + 6
            svg.append(f'<g class="hb">')
            svg.append(f'<rect x="{bx:.1f}" y="{PAD_T}" width="{band_w:.1f}" height="{ph}" '
                       f'fill="transparent" style="cursor:crosshair"/>')
            svg.append(f'<line class="hl" x1="{cx:.1f}" y1="{PAD_T}" x2="{cx:.1f}" y2="{PAD_T+ph}" '
                       f'stroke="rgba(250,250,250,.35)" stroke-width="0.8" stroke-dasharray="4,3"/>')
            svg.append(f'<rect class="hbg" x="{bg_x:.1f}" y="{bg_y}" width="{bg_w:.0f}" height="{bg_h}" '
                       f'fill="#111113" stroke="#3f3f46" stroke-width="0.6" rx="5" ry="5"/>')
            ty_pos = bg_y + 14
            svg.append(f'<text class="ht" x="{cx+tx_off:.1f}" y="{ty_pos}" text-anchor="{t_anchor}" '
                       f'font-family="DM Mono,monospace" font-size="9.5" font-weight="700" fill="#a1a1aa">{html.escape(xv_str)}</text>')
            ty_pos += 4
            for label, val, col in tip_lines:
                ty_pos += 14
                svg.append(f'<text class="ht" x="{cx+tx_off:.1f}" y="{ty_pos:.1f}" text-anchor="{t_anchor}" '
                           f'font-family="DM Mono,monospace" font-size="9" fill="{col}">'
                           f'{html.escape(label)}: {html.escape(val)}</text>')
            svg.append('</g>')

    # Axes borders
    svg.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+ph}" stroke="{GR_SVG}" stroke-width="1"/>')
    svg.append(f'<line x1="{PAD_L}" y1="{PAD_T+ph}" x2="{PAD_L+pw}" y2="{PAD_T+ph}" stroke="{GR_SVG}" stroke-width="1"/>')

    # Labels
    if xlabel:
        svg.append(f'<text x="{PAD_L+pw/2:.0f}" y="{H-4}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="11" font-weight="600" fill="#fafafa">{xlabel}</text>')
    if ylabel:
        svg.append(f'<text transform="rotate(-90)" x="-{PAD_T+ph/2:.0f}" y="12" '
                   f'text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="600" fill="#fafafa">{ylabel}</text>')
    if title:
        svg.append(f'<text x="{W/2:.0f}" y="{PAD_T-8}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="13" font-weight="700" fill="#d4d4d8">{title}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def show_svg(svg_str, height=None, full_width=False):
    fw = "width:100%;" if full_width else ""
    st.markdown(f'<div style="border-radius:10px;overflow:hidden;margin:4px 0;{fw}">'
                f'<div style="{fw}{"" if not full_width else "max-width:100%;overflow-x:auto;"}">{svg_str}</div></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────
def section_header(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def field_label(t): st.markdown(f'<div class="fl">{t}</div>', unsafe_allow_html=True)

def mat_widget(pfx, dy=1, dm=0, dd=0):
    c1,c2,c3 = st.columns(3)
    with c1: y = st.number_input("Annees",0,30,dy,1,key=f"{pfx}_y",help="Annees entieres")
    with c2: m = st.number_input("Mois",  0,11,dm,1,key=f"{pfx}_m",help="Mois supplementaires")
    with c3: d = st.number_input("Jours", 0,30,dd,1,key=f"{pfx}_d",help="Jours supplementaires")
    T = mat_from_ymd(y,m,d)
    return T

def mat_inline(pfx, dy=1, dm=0, dd=0):
    c1,c2,c3 = st.columns(3)
    with c1: y=st.number_input("A",0,30,dy,1,key=f"{pfx}_y",label_visibility="collapsed",help="Annees")
    with c2: m=st.number_input("M",0,11,dm,1,key=f"{pfx}_m",label_visibility="collapsed",help="Mois")
    with c3: d=st.number_input("J",0,30,dd,1,key=f"{pfx}_d",label_visibility="collapsed",help="Jours")
    T = mat_from_ymd(y,m,d)
    return T

def greek_card(sym,name,val,fmt,color,desc):
    st.markdown(f'<div class="gc" style="--acc:{color}"><div style="display:flex;align-items:flex-start;justify-content:space-between">'
                f'<span class="gc-sym">{sym}</span><span class="gc-nm">{name}</span></div>'
                f'<div class="gc-v">{val:{fmt}}</div><div class="gc-d">{desc}</div></div>',
                unsafe_allow_html=True)

def greek_card_html(sym,name,val,fmt,color,desc):
    return (f'<div class="gc" style="--acc:{color}"><div style="display:flex;align-items:flex-start;justify-content:space-between">'
            f'<span class="gc-sym">{sym}</span><span class="gc-nm">{name}</span></div>'
            f'<div class="gc-v">{val:{fmt}}</div><div class="gc-d">{desc}</div></div>')

def signal_card(sc,dc,content):
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
        elif val>0:        return "sg","dg",f"<b>Gamma positif ({val:.5f})</b> — Les gros mouvements jouent en votre faveur. Contrepartie : vous perdez de la valeur chaque jour (Thêta négatif)."
        else:              return "sr","dr",f"<b>Gamma négatif ({val:.5f})</b> — Les gros mouvements vous sont défavorables. Contrepartie : vous encaissez de la valeur chaque jour (Thêta positif)."
    elif name=="theta":
        if abs(val)<0.0005: return "sb","db",f"<b>Theta neutre ({val:+.4f} €/j)</b> — Le temps ne vous impacte pas significativement."
        elif val<0:         return "sr","dr",f"<b>Theta négatif ({val:+.4f} €/j)</b> — Chaque jour qui passe vous coûte {abs(val):.4f} €. Le temps est votre ennemi."
        else:               return "sg","dg",f"<b>Theta positif ({val:+.4f} €/j)</b> — Chaque jour qui passe vous rapporte {val:.4f} €. Le temps est votre allié."
    elif name=="vega":
        if abs(val)<0.01: return "sb","db",f"<b>Vega faible ({val:.4f} €/%)</b> — La volatilité du marche vous affecte peu."
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
#  HELPER FUNCTIONS — Strategy leg, pré-expiration, probabilité
# ─────────────────────────────────────────────────────────────
def get_strategy_legs(name, K, offset):
    """Return list of (direction_qty, instrument, strike) for pre-expiration repricing."""
    Ko_c, Ko_p = K + offset, K - offset
    Kl, Kh = K - offset, K + offset
    Kic_l, Kic_h = K - 2*offset, K + 2*offset
    M = {
        "Long Straddle":   [(1,"call",K),(1,"put",K)],
        "Short Straddle":  [(-1,"call",K),(-1,"put",K)],
        "Long Strangle":   [(1,"call",Ko_c),(1,"put",Ko_p)],
        "Short Strangle":  [(-1,"call",Ko_c),(-1,"put",Ko_p)],
        "Bull Call Spread": [(1,"call",K),(-1,"call",Kh)],
        "Bear Put Spread":  [(1,"put",Kh),(-1,"put",K)],
        "Long Butterfly":   [(1,"call",Kl),(-2,"call",K),(1,"call",Kh)],
        "Short Butterfly":  [(-1,"call",Kl),(2,"call",K),(-1,"call",Kh)],
        "Iron Condor":     [(1,"put",Kic_l),(-1,"put",Ko_p),(-1,"call",Ko_c),(1,"call",Kic_h)],
        "Iron Butterfly":  [(1,"put",Kl),(-1,"put",K),(-1,"call",K),(1,"call",Kh)],
        "Long Call":       [(1,"call",K)],
        "Long Put":        [(1,"put",K)],
        "Covered Call":    [(1,"stock",0),(-1,"call",Kh)],
        "Protective Put":  [(1,"stock",0),(1,"put",Kl)],
        "Bull Put Spread":  [(-1,"put",K),(1,"put",Kl)],
        "Bear Call Spread": [(-1,"call",K),(1,"call",Kh)],
        "Long Call Spread 1x2": [(1,"call",K),(-2,"call",Kh)],
        "Short Call":       [(-1,"call",K)],
        "Short Put":        [(-1,"put",K)],
        "Synthetic Long Forward":  [(1,"call",K),(-1,"put",K)],
        "Synthetic Short Forward": [(-1,"call",K),(1,"put",K)],
    }
    return M.get(name, [])

def compute_pre_exp_pnl(legs, SR, S, Tc, Tp, r, sc, sp, q, T_pct):
    """Compute P&L at T_pct fraction of time elapsed."""
    T_rem_c = max(Tc * (1 - T_pct), 1e-9)
    T_rem_p = max(Tp * (1 - T_pct), 1e-9)
    cost = 0.0
    for d, inst, k in legs:
        if inst == "stock": cost += d * S
        elif inst == "call": cost += d * bs_price(S, k, Tc, r, sc, q, "call")
        else: cost += d * bs_price(S, k, Tp, r, sp, q, "put")
    total = np.zeros_like(SR, dtype=float)
    for d, inst, k in legs:
        if inst == "stock": total += d * SR
        elif inst == "call": total += d * bs_price_vec(SR, k, T_rem_c, r, sc, q, "call")
        else: total += d * bs_price_vec(SR, k, T_rem_p, r, sp, q, "put")
    return total - cost

def probability_of_profit(pnl, SR, S, T, r, sigma, q=0.0):
    """P(profit) using log-normal distribution on spot grid."""
    if T <= 1e-9 or sigma <= 1e-9 or len(SR) < 10: return 0.5
    mu = (r - q - 0.5*sigma**2) * T
    sig_t = sigma * np.sqrt(T)
    SR_pos = np.maximum(SR, 1e-10)
    pdf_vals = norm.pdf(np.log(SR_pos/S), mu, sig_t) / SR_pos
    dx = np.diff(SR)
    pdf_mid = (pdf_vals[:-1] + pdf_vals[1:]) / 2
    pnl_mid = (pnl[:-1] + pnl[1:]) / 2
    total = np.sum(pdf_mid * dx)
    if total < 1e-12: return 0.5
    return float(np.sum(pdf_mid[pnl_mid > 0] * dx[pnl_mid > 0]) / total)

def scenario_grid_html(S, K, T, r, sigma, q, otype, pos_sign):
    spot_pcts = [-10, -5, 0, +5, +10]
    vol_pps   = [-10, -5, 0, +5, +10]
    base = bs_price(S, K, T, r, sigma, q, otype) * pos_sign
    h = '<table class="sc-grid"><tr><th style="min-width:80px">Spot \\ \u03c3</th>'
    for dv in vol_pps:
        lbl = f"\u03c3 {dv:+d}pp" if dv != 0 else "\u03c3 actuel"
        h += f'<th>{lbl}</th>'
    h += '</tr>'
    for ds in spot_pcts:
        S_n = S * (1 + ds/100)
        rl = f"S {ds:+d}%" if ds != 0 else "S actuel"
        h += f'<tr><td style="font-weight:700;color:var(--t2);background:var(--s2);text-align:left;padding-left:12px">{rl}<br><span style="font-size:.68rem;color:var(--t3)">{S_n:.1f} \u20ac</span></td>'
        for dv in vol_pps:
            sig_n = max(sigma + dv/100, 0.01)
            p = bs_price(S_n, K, T, r, sig_n, q, otype) * pos_sign
            diff = p - base
            is_ref = ds == 0 and dv == 0
            cls = "sc-ref" if is_ref else ("sc-pos" if diff > 0.0001 else ("sc-neg" if diff < -0.0001 else "sc-zero"))
            diff_str = f'<span style="font-size:.64rem;opacity:.7">({diff:+.4f})</span>' if not is_ref else '<span style="font-size:.64rem;opacity:.7">ref</span>'
            h += f'<td class="{cls}">{p:.4f}<br>{diff_str}</td>'
        h += '</tr>'
    h += '</table>'
    return h

def risk_reward_html(max_gain, max_loss):
    """Beautiful risk/reward card."""
    if abs(max_loss) < 1e-6: ratio_str = "\u221e"
    else: ratio_str = f"{abs(max_gain/max_loss):.2f}"
    total = abs(max_gain) + abs(max_loss)
    gain_pct = abs(max_gain) / total * 100 if total > 0 else 50
    return (f'<div class="rr"><div class="rr-top"><span class="rr-label">Risk / Reward</span>'
            f'<span class="rr-ratio">{ratio_str}</span></div>'
            f'<div class="rr-bar"><div class="rr-gain" style="width:{gain_pct:.1f}%"></div>'
            f'<div class="rr-loss" style="width:{100-gain_pct:.1f}%"></div></div>'
            f'<div class="rr-vals"><span style="color:#22c55e">+\u20ac{abs(max_gain):.2f}</span>'
            f'<span style="color:#ef4444">\u2212\u20ac{abs(max_loss):.2f}</span></div></div>')

def prob_bar_html(p_profit, label="Probabilité de profit"):
    """Probability bar visualization."""
    pct = p_profit * 100
    col = "#22c55e" if pct >= 55 else ("#f59e0b" if pct >= 45 else "#ef4444")
    sentiment = "Favorable" if pct >= 55 else ("Neutre" if pct >= 45 else "Défavorable")
    return (f'<div class="prob"><div class="prob-top"><span class="prob-label">{label}</span>'
            f'<span class="prob-val" style="color:{col}">{pct:.1f}%</span></div>'
            f'<div class="prob-bar"><div class="prob-fill" style="width:{pct:.1f}%;background:{col}"></div></div>'
            f'<div class="rr-vals"><span style="color:{col}">{sentiment}</span>'
            f'<span style="color:var(--t3)">{100-pct:.1f}% de perte</span></div></div>')

# ─────────────────────────────────────────────────────────────
#  DASHBOARD — 5 SVG charts
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_dashboard(S, K, T, r, sigma, q, otype, pos_sign=1):
    N = 200
    SR   = np.linspace(max(S*0.45,1), S*1.55, N)
    sigR = np.linspace(0.02, 1.0, N)
    TR   = np.linspace(0.005, max(T*2.5,0.5), N)

    prices = np.array([bs_price(s,K,T,r,sigma,q,otype) for s in SR]) * pos_sign
    gs_all = [bs_greeks(s,K,T,r,sigma,q,otype) for s in SR]
    deltas = np.array([g["delta"] for g in gs_all]) * pos_sign
    gammas = np.array([g["gamma"] for g in gs_all]) * pos_sign
    intr   = np.array([max(s-K,0) if otype=="call" else max(K-s,0) for s in SR]) * pos_sign
    p_sig  = np.array([bs_price(S,K,T,r,s,q,otype) for s in sigR]) * pos_sign
    p_T    = np.array([bs_price(S,K,t,r,sigma,q,otype) for t in TR]) * pos_sign
    cur    = bs_price(S,K,T,r,sigma,q,otype) * pos_sign
    G      = {k: v * pos_sign for k, v in bs_greeks(S,K,T,r,sigma,q,otype).items()}

    W, H = 680, 260

    def vl(xv, col, lbl="", dash=True):
        return {"x":xv,"color":col,"label":lbl,"dash":dash}

    _rsp = True  # responsive SVGs

    # 1 Prix
    svg1 = svg_chart([
        {"x":list(SR),"y":list(intr),"color":"#52525b","width":1,"dash":True,"label":"Valeur intrinsèque"},
        {"x":list(SR),"y":list(prices),"color":"#f59e0b","width":2.2,"fill":True,"label":f"Prix {otype.upper()}"},
    ], W=W, H=H, xlabel="Spot (\u20ac)", ylabel="Prix (\u20ac)",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b",f"K={K:.0f}")],
       show_dot={"x":S,"y":cur,"color":"#f59e0b","label":f"\u20ac{cur:.3f}"},
       title="Prix de l'option selon le spot", responsive=_rsp)

    # 2 Delta
    svg2 = svg_chart([
        {"x":list(SR),"y":list(deltas),"color":"#22c55e","width":2.2,"fill":True,"fill_color":"#22c55e","label":"\u0394 Delta"},
    ], W=W, H=H, xlabel="Spot (\u20ac)", ylabel="Delta",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b")],
       hline_zero=True,
       show_dot={"x":S,"y":G["delta"],"color":"#22c55e","label":f"{G['delta']:.4f}"},
       title="\u0394 Delta - Sensibilité au prix du sous-jacent", responsive=_rsp)

    # 3 Gamma
    svg3 = svg_chart([
        {"x":list(SR),"y":list(gammas),"color":"#a78bfa","width":2.2,"fill":True,"fill_color":"#a78bfa","label":"\u0393 Gamma"},
    ], W=W, H=H, xlabel="Spot (\u20ac)", ylabel="Gamma",
       vlines=[vl(S,"#3b82f6",f"S={S:.0f}"), vl(K,"#52525b")],
       hline_zero=True,
       show_dot={"x":S,"y":G["gamma"],"color":"#a78bfa","label":f"{G['gamma']:.5f}"},
       title="\u0393 Gamma - Convexité (accélération du Delta)", responsive=_rsp)

    # 4 Prix vs Vol
    svg4 = svg_chart([
        {"x":list(sigR*100),"y":list(p_sig),"color":"#ef4444","width":2.2,"fill":True,"fill_color":"#ef4444","label":"Prix"},
    ], W=W, H=H, xlabel="Volatilité implicite (%)", ylabel="Prix (\u20ac)",
       vlines=[vl(sigma*100,"#f59e0b",f"\u03c3={sigma*100:.1f}%")],
       show_dot={"x":sigma*100,"y":cur,"color":"#f59e0b","label":f"\u20ac{cur:.3f}"},
       title="\u03bd Vega - Sensibilité à la volatilité implicite", responsive=_rsp)

    # 5 Time Decay
    svg5 = svg_chart([
        {"x":list(TR),"y":list(p_T),"color":"#3b82f6","width":2.2,"fill":True,"fill_color":"#3b82f6","label":"Prix"},
    ], W=W, H=H, xlabel="Maturité (ans)", ylabel="Prix (\u20ac)",
       vlines=[vl(T,"#f59e0b",fmt_mat(T))],
       show_dot={"x":T,"y":cur,"color":"#f59e0b","label":f"\u20ac{cur:.3f}"},
       title="\u0398 Theta - \u00c9rosion temporelle (Time Decay)", responsive=_rsp)

    return svg1,svg2,svg3,svg4,svg5

# ─────────────────────────────────────────────────────────────
#  PAYOFF CHART
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_payoff(name, S, K, Tc, Tp, r, sc, sp, q=0.0, T_pct=0.5, W=1100, H=380):
    SR = np.linspace(S*0.50, S*1.50, 400)
    strike_offset = S * 0.07
    call_px = lambda k, T, sg: bs_price(S, k, T, r, sg, q, "call")
    put_px  = lambda k, T, sg: bs_price(S, k, T, r, sg, q, "put")
    C0 = call_px(K, Tc, sc);  P0 = put_px(K, Tp, sp)
    K_call_otm, K_put_otm = K + strike_offset, K - strike_offset
    K_low_wing, K_high_wing = K - strike_offset, K + strike_offset
    C_low = call_px(K_low_wing, Tc, sc);  C_high = call_px(K_high_wing, Tc, sc)
    K_ic_low, K_ic_mid_low, K_ic_mid_high, K_ic_high = K - 2*strike_offset, K - strike_offset, K + strike_offset, K + 2*strike_offset
    P_ic_low = put_px(K_ic_low, Tp, sp);  P_ic_mid = put_px(K_ic_mid_low, Tp, sp)
    C_ic_mid = call_px(K_ic_mid_high, Tc, sc);  C_ic_high = call_px(K_ic_high, Tc, sc)

    mx = np.maximum
    pnls = {
        "Long Straddle":   mx(SR-K,0)+mx(K-SR,0)-C0-P0,
        "Short Straddle":  -(mx(SR-K,0)+mx(K-SR,0))+C0+P0,
        "Long Strangle":   mx(SR-K_call_otm,0)+mx(K_put_otm-SR,0)-call_px(K_call_otm,Tc,sc)-put_px(K_put_otm,Tp,sp),
        "Short Strangle":  -(mx(SR-K_call_otm,0)+mx(K_put_otm-SR,0))+call_px(K_call_otm,Tc,sc)+put_px(K_put_otm,Tp,sp),
        "Bull Call Spread": mx(SR-K,0)-mx(SR-K_high_wing,0)-(C0-C_high),
        "Bear Put Spread":  mx(K_high_wing-SR,0)-mx(K-SR,0)-(put_px(K_high_wing,Tp,sp)-P0),
        "Long Butterfly":   mx(SR-K_low_wing,0)-2*mx(SR-K,0)+mx(SR-K_high_wing,0)-(C_low-2*C0+C_high),
        "Short Butterfly":  -(mx(SR-K_low_wing,0)-2*mx(SR-K,0)+mx(SR-K_high_wing,0))+(C_low-2*C0+C_high),
        "Iron Condor":     -mx(K_ic_mid_low-SR,0)+mx(K_ic_low-SR,0)-mx(SR-K_ic_mid_high,0)+mx(SR-K_ic_high,0)+(P_ic_mid-P_ic_low+C_ic_mid-C_ic_high),
        "Iron Butterfly":  -mx(SR-K,0)-mx(K-SR,0)+mx(SR-K_high_wing,0)+mx(K_low_wing-SR,0)+(-C0-P0+C_high+put_px(K_low_wing,Tp,sp)),
        "Long Call":       mx(SR-K,0)-C0,
        "Long Put":        mx(K-SR,0)-P0,
        "Covered Call":    (SR-S)-mx(SR-K_high_wing,0)+call_px(K_high_wing,Tc,sc),
        "Protective Put":  (SR-S)+mx(K_low_wing-SR,0)-put_px(K_low_wing,Tp,sp),
        "Bull Put Spread":  -(mx(K-SR,0)-mx(K_low_wing-SR,0))+(P0-put_px(K_low_wing,Tp,sp)),
        "Bear Call Spread": -(mx(SR-K,0)-mx(SR-K_high_wing,0))+(C0-C_high),
        "Long Call Spread 1x2": mx(SR-K,0)-2*mx(SR-K_high_wing,0)-(C0-2*C_high),
        "Short Call":       -mx(SR-K,0)+C0,
        "Short Put":        -mx(K-SR,0)+P0,
        "Synthetic Long Forward":  mx(SR-K,0)-mx(K-SR,0)-(C0-P0),
        "Synthetic Short Forward": -mx(SR-K,0)+mx(K-SR,0)+(C0-P0),
    }
    pnl = pnls.get(name, np.zeros_like(SR))
    col = STRATEGIES[name]["color"]

    # Pre-expiration P&L
    offset = S * 0.07
    legs = get_strategy_legs(name, K, offset)
    pnl_pre = compute_pre_exp_pnl(legs, SR, S, Tc, Tp, r, sc, sp, q, T_pct)
    T_avg = (Tc + Tp) / 2
    T_rem = T_avg * (1 - T_pct)
    time_label = f"P&L à {T_pct*100:.0f}% ({fmt_mat(T_rem)} restant)"

    vlines = [{"x":S,"color":"#3b82f6","label":f"S={S:.0f}","dash":True},
               {"x":K,"color":"#52525b","label":f"K={K:.0f}","dash":True}]
    # break-evens as vlines
    idxs = np.where(np.diff(np.sign(pnl)))[0]
    for idx in idxs:
        be = (SR[idx]+SR[idx+1])/2
        vlines.append({"x":be,"color":"#f59e0b","label":f"BE {be:.1f}","dash":True})

    legend_items = [
        {"label":time_label,"color":"#ffffff","dash":False},
        {"label":f"S\u2080 = {S:.0f}","color":"#3b82f6","dash":True},
        {"label":f"K = {K:.0f}","color":"#52525b","dash":True},
    ]
    for vl in vlines:
        if vl.get("label","").startswith("BE"):
            legend_items.append({"label":vl["label"],"color":vl["color"],"dash":True})

    svg = svg_chart([
        {"x":list(SR),"y":list(pnl_pre),"color":"#ffffff","width":3,"fill_pos_neg":True},
    ], W=W, H=H, xlabel="Prix du sous-jacent (\u20ac)", ylabel="P&L (\u20ac)",
       hline_zero=True, vlines=vlines,
       title=f"{name} - Profit / Perte",
       PAD_L=60, PAD_R=24, PAD_T=30, PAD_B=46,
       responsive=True)
    return svg, legend_items, pnl, pnl_pre

# ─────────────────────────────────────────────────────────────
#  CUSTOM PAYOFF + GREEKS CHART
# ─────────────────────────────────────────────────────────────
PALETTE = ["#3b82f6","#22c55e","#f59e0b","#a78bfa","#ef4444","#06b6d4"]

@st.cache_data(show_spinner=False)
def build_custom_payoff(legs, S_ref, name, T_pct=0.5, W=1100, H=420):
    SR = np.linspace(S_ref*0.5, S_ref*1.5, 400)
    total = np.zeros_like(SR); net_prem=0.0; series=[]; legend_items=[]
    for i,leg in enumerate(legs):
        if not leg["active"]: continue
        pr   = bs_price(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],leg.get("q",0),leg["inst"])
        cost = leg["dir"]*pr*leg["qty"]; net_prem+=cost
        pnl  = (leg["dir"]*np.maximum(SR-leg["K"],0)*leg["qty"]-cost
                if leg["inst"]=="call"
                else leg["dir"]*np.maximum(leg["K"]-SR,0)*leg["qty"]-cost)
        total += pnl
        short_dir = "Achat" if leg["dir"] == 1 else "Vente"
        short_label = f"L{i+1} \u00b7 {short_dir} {leg['inst'].upper()} K={leg['K']:.0f}"
        color = PALETTE[i%6]
        series.append({"x":list(SR),"y":list(pnl),"color":color,
                       "width":1.5,"dash":True})
        legend_items.append({"label":short_label,"color":color,"dash":True})

    # Pre-expiration P&L
    total_pre = np.zeros_like(SR)
    for i, leg in enumerate(legs):
        if not leg["active"]: continue
        pr = bs_price(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],leg.get("q",0),leg["inst"])
        cost_leg = leg["dir"]*pr*leg["qty"]
        T_rem = max(leg["T"] * (1 - T_pct), 1e-9)
        vals = bs_price_vec(SR, leg["K"], T_rem, leg["r"], leg["sigma"], leg.get("q",0), leg["inst"])
        total_pre += leg["dir"]*leg["qty"]*vals - cost_leg

    # Compute average T for label
    active_Ts = [l["T"] for l in legs if l.get("active")]
    T_avg = sum(active_Ts)/len(active_Ts) if active_Ts else 1.0
    T_rem_lbl = T_avg * (1 - T_pct)
    pre_label = f"P&L à {T_pct*100:.0f}% ({fmt_mat(T_rem_lbl)} restant)"

    series.append({"x":list(SR),"y":list(total_pre),"color":"#ffffff","width":3,
                   "fill_pos_neg":True})
    legend_items.append({"label":pre_label,"color":"#ffffff","dash":False})

    vlines=[{"x":S_ref,"color":"#f472b6","dash":True}]
    legend_items.append({"label":f"S\u2080 = {S_ref:.0f}","color":"#f472b6","dash":True})
    idxs=np.where(np.diff(np.sign(total)))[0]
    for idx in idxs:
        be=(SR[idx]+SR[idx+1])/2
        vlines.append({"x":be,"color":"#e2e8f0","dash":True})
        legend_items.append({"label":f"BE {be:.1f}","color":"#e2e8f0","dash":True})

    nc="encaissé" if net_prem<0 else "payé"
    svg = svg_chart(series, W=W, H=H,
                    xlabel="Prix à l'expiration (\u20ac)", ylabel="P&L (\u20ac)",
                    hline_zero=True, vlines=vlines,
                    title=f"{name}  \u00b7  Prime nette {nc} : \u20ac{abs(net_prem):.4f}",
                    PAD_L=62, PAD_R=24, PAD_T=20, PAD_B=48,
                    responsive=True)
    return svg, total, net_prem, legend_items, total_pre

@st.cache_data(show_spinner=False)
def build_custom_greeks(legs, S_ref, W=1100, H=260):
    SR=np.linspace(S_ref*0.5,S_ref*1.5,150)
    PD=np.zeros_like(SR); PG=np.zeros_like(SR)
    PT=np.zeros_like(SR); PV=np.zeros_like(SR)
    for leg in legs:
        if not leg["active"]: continue
        for i,s in enumerate(SR):
            G=bs_greeks(s,leg["K"],leg["T"],leg["r"],leg["sigma"],leg.get("q",0),leg["inst"])
            PD[i]+=leg["dir"]*leg["qty"]*G["delta"]
            PG[i]+=leg["dir"]*leg["qty"]*G["gamma"]
            PT[i]+=leg["dir"]*leg["qty"]*G["theta"]
            PV[i]+=leg["dir"]*leg["qty"]*G["vega"]

    vl=[{"x":S_ref,"color":"#3b82f6","dash":True}]
    kwargs=dict(W=W//4, H=H, hline_zero=True, vlines=vl,
                xlabel="Spot (\u20ac)", PAD_L=48,PAD_R=12,PAD_T=30,PAD_B=36,
                responsive=True)

    svgs=[]
    for data,col,title in [(PD,"#22c55e","\u0394 Delta - Sensibilité prix"),(PG,"#a78bfa","\u0393 Gamma - Convexité"),
                            (PT,"#f59e0b","\u0398 Theta - Effet temps"),(PV,"#3b82f6","\u03bd Vega - Effet volatilité")]:
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
     "max_gain":"Illimité","max_loss":"Prime totale","be":"K \u00b1 prime",
     "greeks":"Long Gamma \u00b7 Short Theta \u00b7 Long Vega \u00b7 Delta \u2248 0","color":"#22c55e"},
    "Short Straddle":  {"desc":"Vendre call + put au même strike. Encaisse la prime si le marché reste stable.",
     "legs":[("sell","call","K ATM"),("sell","put","K ATM")],"outlook":"Marché calme attendu",
     "max_gain":"Prime totale","max_loss":"Illimité","be":"K \u00b1 prime",
     "greeks":"Short Gamma \u00b7 Long Theta \u00b7 Short Vega \u00b7 Delta \u2248 0","color":"#ef4444"},
    "Long Strangle":   {"desc":"Acheter call OTM + put OTM. Moins cher que le straddle, mais le marché doit bouger encore plus.",
     "legs":[("buy","call","K+7%"),("buy","put","K\u22127%")],"outlook":"Très grand mouvement",
     "max_gain":"Illimité","max_loss":"Prime totale","be":"K\u00b1 + prime",
     "greeks":"Long Gamma \u00b7 Short Theta \u00b7 Long Vega \u00b7 Delta \u2248 0","color":"#3b82f6"},
    "Short Strangle":  {"desc":"Vendre call OTM + put OTM. Zone de gain plus large que le straddle court.",
     "legs":[("sell","call","K+7%"),("sell","put","K\u22127%")],"outlook":"Marché range",
     "max_gain":"Prime totale","max_loss":"Illimité","be":"K\u00b1 + prime",
     "greeks":"Short Gamma \u00b7 Long Theta \u00b7 Short Vega","color":"#f59e0b"},
    "Bull Call Spread": {"desc":"Acheter un call, vendre un call plus élevé. Exposition haussière limité à moindre co\u00fbt.",
     "legs":[("buy","call","K1"),("sell","call","K2>K1")],"outlook":"Haussier modéré",
     "max_gain":"(K2\u2212K1)\u2212prime","max_loss":"Prime nette","be":"K1 + prime",
     "greeks":"Delta + \u00b7 Gamma faible \u00b7 Theta neutre \u00b7 Vega faible","color":"#22c55e"},
    "Bear Put Spread":  {"desc":"Acheter un put, vendre un put inférieur. Exposition baissière limité à moindre co\u00fbt.",
     "legs":[("buy","put","K2"),("sell","put","K1<K2")],"outlook":"Baissier modéré",
     "max_gain":"(K2\u2212K1)\u2212prime","max_loss":"Prime nette","be":"K2 \u2212 prime",
     "greeks":"Delta \u2212 \u00b7 Gamma faible \u00b7 Theta neutre \u00b7 Vega faible","color":"#a78bfa"},
    "Long Butterfly":  {"desc":"Acheter les ailes K1 et K3, vendre 2\u00d7 le corps K2. Gagne si le prix expire exactement à K2.",
     "legs":[("buy","call","K1"),("sell","call","K2 \u00d72"),("buy","call","K3")],"outlook":"Marché très stable",
     "max_gain":"(K2\u2212K1)\u2212prime","max_loss":"Prime nette","be":"K1+prime \u00b7 K3\u2212prime",
     "greeks":"Delta \u2248 0 \u00b7 Short Gamma \u00b7 Long Theta \u00b7 Vega faible","color":"#f59e0b"},
    "Short Butterfly": {"desc":"Inverse du butterfly. Vendre les ailes, acheter le corps. Gagne si le prix s'éloigne de K2.",
     "legs":[("sell","call","K1"),("buy","call","K2 \u00d72"),("sell","call","K3")],"outlook":"Mouvement attendu",
     "max_gain":"Prime nette","max_loss":"(K2\u2212K1)\u2212prime","be":"K1+prime \u00b7 K3\u2212prime",
     "greeks":"Long Gamma \u00b7 Short Theta \u00b7 Vega faible","color":"#ef4444"},
    "Iron Condor":     {"desc":"Vendre put spread + call spread. Gagne si le prix reste dans le couloir à l'expiration.",
     "legs":[("buy","put","K1"),("sell","put","K2"),("sell","call","K3"),("buy","call","K4")],"outlook":"Marché range \u00b7 Risque limité",
     "max_gain":"Prime nette","max_loss":"Largeur \u2212 prime","be":"K2\u2212prime \u00b7 K3+prime",
     "greeks":"Delta \u2248 0 \u00b7 Short Gamma \u00b7 Long Theta \u00b7 Short Vega","color":"#3b82f6"},
    "Iron Butterfly":  {"desc":"Vendre straddle ATM + acheter strangle OTM. Gain max si expiration au strike central.",
     "legs":[("buy","put","K1"),("sell","put","K ATM"),("sell","call","K ATM"),("buy","call","K3")],"outlook":"Très stable",
     "max_gain":"Prime nette","max_loss":"Largeur \u2212 prime","be":"K \u00b1 prime",
     "greeks":"Delta \u2248 0 \u00b7 Short Gamma fort \u00b7 Long Theta \u00b7 Short Vega","color":"#f59e0b"},
    "Long Call":       {"desc":"Droit d'acheter à prix fixe. Gain illimité si le prix monte, perte limité à la prime.",
     "legs":[("buy","call","K")],"outlook":"Haussier",
     "max_gain":"Illimité","max_loss":"Prime payé","be":"K + prime",
     "greeks":"Delta + \u00b7 Gamma + \u00b7 Theta \u2212 \u00b7 Vega +","color":"#22c55e"},
    "Long Put":        {"desc":"Droit de vendre à prix fixe. Sert de protection ou d'exposition baissière.",
     "legs":[("buy","put","K")],"outlook":"Baissier ou protection",
     "max_gain":"K \u2212 prime","max_loss":"Prime payé","be":"K \u2212 prime",
     "greeks":"Delta \u2212 \u00b7 Gamma + \u00b7 Theta \u2212 \u00b7 Vega +","color":"#a78bfa"},
    "Covered Call":    {"desc":"Détenir l'action et vendre un call. Génère un revenu, plafonne le gain à la hausse.",
     "legs":[("buy","stock","actions"),("sell","call","K>S")],"outlook":"Neutre \u00b7 Revenu",
     "max_gain":"(K\u2212S)+prime","max_loss":"S\u2212prime","be":"S \u2212 prime",
     "greeks":"Delta réduit \u00b7 Short Gamma \u00b7 Long Theta \u00b7 Short Vega","color":"#f59e0b"},
    "Protective Put":  {"desc":"Détenir l'action et acheter un put. Assurance contre une baisse, gain illimité à la hausse.",
     "legs":[("buy","stock","actions"),("buy","put","K")],"outlook":"Haussier avec protection",
     "max_gain":"Illimité","max_loss":"S\u2212K+prime","be":"S + prime",
     "greeks":"Delta + \u00b7 Gamma + \u00b7 Theta \u2212 \u00b7 Vega +","color":"#3b82f6"},
    "Bull Put Spread":  {"desc":"Vendre un put ATM, acheter un put OTM. Crédit re\u00e7u, exposition haussière à risque limité.",
     "legs":[("sell","put","K2"),("buy","put","K1<K2")],"outlook":"Haussier modéré",
     "max_gain":"Prime nette","max_loss":"(K2\u2212K1)\u2212prime","be":"K2 \u2212 prime",
     "greeks":"Delta + \u00b7 Gamma faible \u00b7 Theta + \u00b7 Short Vega","color":"#22c55e"},
    "Bear Call Spread":  {"desc":"Vendre un call ATM, acheter un call OTM. Crédit re\u00e7u, exposition baissière à risque limité.",
     "legs":[("sell","call","K1"),("buy","call","K2>K1")],"outlook":"Baissier modéré",
     "max_gain":"Prime nette","max_loss":"(K2\u2212K1)\u2212prime","be":"K1 + prime",
     "greeks":"Delta \u2212 \u00b7 Gamma faible \u00b7 Theta + \u00b7 Short Vega","color":"#a78bfa"},
    "Long Call Spread 1x2": {"desc":"Acheter 1 call ATM, vendre 2 calls OTM. Financement partiel, risque à la hausse illimité.",
     "legs":[("buy","call","K1"),("sell","call","K2>K1 \u00d72")],"outlook":"Haussier modéré",
     "max_gain":"(K2\u2212K1)\u2212prime","max_loss":"Illimité au-dessus","be":"K1+prime \u00b7 2K2\u2212K1\u2212prime",
     "greeks":"Delta + \u00b7 Long puis Short Gamma \u00b7 Theta mixte","color":"#06b6d4"},
    "Short Call":       {"desc":"Vendre un call nu. Encaisse la prime, risque illimité à la hausse.",
     "legs":[("sell","call","K")],"outlook":"Baissier ou neutre",
     "max_gain":"Prime encaissé","max_loss":"Illimité","be":"K + prime",
     "greeks":"Delta \u2212 \u00b7 Gamma \u2212 \u00b7 Theta + \u00b7 Vega \u2212","color":"#ef4444"},
    "Short Put":        {"desc":"Vendre un put nu. Encaisse la prime, risque si le marché chute fortement.",
     "legs":[("sell","put","K")],"outlook":"Haussier ou neutre",
     "max_gain":"Prime encaissé","max_loss":"K \u2212 prime","be":"K \u2212 prime",
     "greeks":"Delta + \u00b7 Gamma \u2212 \u00b7 Theta + \u00b7 Vega \u2212","color":"#f59e0b"},
    "Synthetic Long Forward":  {"desc":"Acheter call + vendre put au même strike. Reproduit un contrat forward long via la parité put-call (C \u2212 P = S \u2212 K\u00b7e\u207b\u02b3\u1d40). Aucune prime nette en théorie si K = forward.",
     "legs":[("buy","call","K ATM"),("sell","put","K ATM")],"outlook":"Haussier \u00b7 Réplication forward",
     "max_gain":"Illimité","max_loss":"K \u2212 prime nette","be":"K + prime nette",
     "greeks":"Delta \u2248 +1 \u00b7 Gamma \u2248 0 \u00b7 Theta \u2248 0 \u00b7 Vega \u2248 0","color":"#22c55e"},
    "Synthetic Short Forward": {"desc":"Vendre call + acheter put au même strike. Reproduit un forward short via la parité put-call. \u00c9quivalent économique d'une vente à découvert du sous-jacent.",
     "legs":[("sell","call","K ATM"),("buy","put","K ATM")],"outlook":"Baissier \u00b7 Réplication forward",
     "max_gain":"K \u2212 prime nette","max_loss":"Illimité","be":"K \u2212 prime nette",
     "greeks":"Delta \u2248 \u22121 \u00b7 Gamma \u2248 0 \u00b7 Theta \u2248 0 \u00b7 Vega \u2248 0","color":"#ef4444"},
}
LEG_A={"buy":("tby","Achat"),"sell":("tse","Vente")}
LEG_I={"call":("tca","Call"),"put":("tpu","Put"),"stock":("tca","Action")}

def legs_html(legs):
    h=""
    for a,ins,strike in legs:
        ac,at=LEG_A.get(a,("tby",a)); ic,it=LEG_I.get(ins,("tca",ins))
        h+=f'<span class="tag {ac}">{at}</span><span class="tag {ic}">{it}</span>'
        h+=f'<span style="font-size:.68rem;color:#a1a1aa;margin:0 3px">{strike}</span>'
    return h

# ─────────────────────────────────────────────────────────────
#  BUILDER TEMPLATES
# ─────────────────────────────────────────────────────────────
BUILDER_TEMPLATES = {
    "Long Straddle":    {"n":2, "legs":[
        dict(dir=1, inst="call", K_pct=0),
        dict(dir=1, inst="put",  K_pct=0)]},
    "Short Straddle":   {"n":2, "legs":[
        dict(dir=-1, inst="call", K_pct=0),
        dict(dir=-1, inst="put",  K_pct=0)]},
    "Bull Call Spread":  {"n":2, "legs":[
        dict(dir=1,  inst="call", K_pct=0),
        dict(dir=-1, inst="call", K_pct=0.07)]},
    "Bear Put Spread":   {"n":2, "legs":[
        dict(dir=1,  inst="put",  K_pct=0.07),
        dict(dir=-1, inst="put",  K_pct=0)]},
    "Iron Condor":       {"n":4, "legs":[
        dict(dir=1,  inst="put",  K_pct=-0.14),
        dict(dir=-1, inst="put",  K_pct=-0.07),
        dict(dir=-1, inst="call", K_pct=0.07),
        dict(dir=1,  inst="call", K_pct=0.14)]},
    "Long Butterfly":    {"n":3, "legs":[
        dict(dir=1,  inst="call", K_pct=-0.07, qty=1),
        dict(dir=-1, inst="call", K_pct=0, qty=2),
        dict(dir=1,  inst="call", K_pct=0.07, qty=1)]},
    "Covered Call":      {"n":2, "legs":[
        dict(dir=1,  inst="call", K_pct=0),
        dict(dir=-1, inst="call", K_pct=0.07)]},
    "Protective Put":    {"n":2, "legs":[
        dict(dir=1,  inst="put",  K_pct=-0.07),
        dict(dir=1,  inst="call", K_pct=0)]},
}

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-l">
    <div class="hdr-ico">\u25c8</div>
    <div><div class="hdr-t">Options Lab</div><div class="hdr-s">Pricer Black-Scholes \u00b7 Grecques \u00b7 Stratégies</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SIDEBAR — Paramètres Pricer
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">\u25c8 Options Lab - Paramètres</div>', unsafe_allow_html=True)

    # ── Type d'option ───────────────────────────────────────
    st.markdown('<div class="sb-title">Type d\'option</div>', unsafe_allow_html=True)
    otype = st.radio("", ["Call", "Put"], horizontal=True, key="ot1",
                     label_visibility="collapsed")
    _badge_cls = "sb-Call" if otype == "Call" else "sb-Put"
    _badge_lbl = "\u25cf CALL - droit d'acheter" if otype == "Call" else "\u25cf PUT - droit de vendre"
    st.markdown(f'<div class="sb-badge {_badge_cls}">{_badge_lbl}</div>', unsafe_allow_html=True)

    # ── Position (Achat / Vente) ───────────────────────────
    st.markdown('<div class="sb-title">Position</div>', unsafe_allow_html=True)
    position = st.radio("", ["Achat (Long)", "Vente (Short)"], horizontal=True, key="pos1",
                        label_visibility="collapsed")
    pos_sign = 1 if position == "Achat (Long)" else -1
    _pos_cls = "sb-Call" if pos_sign == 1 else "sb-Put"
    _pos_lbl = "\u25cf LONG - acheteur de l'option" if pos_sign == 1 else "\u25cf SHORT - vendeur de l'option"
    st.markdown(f'<div class="sb-badge {_pos_cls}">{_pos_lbl}</div>', unsafe_allow_html=True)

    # ── Sous-jacent & Strike ────────────────────────────────
    st.markdown('<div class="sb-title">Sous-jacent & Strike</div>', unsafe_allow_html=True)
    field_label("Spot S\u2080  (\u20ac)")
    S = st.number_input("S0", value=st.session_state.get("shared_S", 100.0),
                        step=1.0, label_visibility="collapsed",
                        help="Prix actuel de l'actif sur le marché", key="t1_S")
    st.session_state["shared_S"] = S

    field_label("Strike K  (\u20ac)")
    K = st.number_input("K", value=st.session_state.get("shared_K", 100.0),
                        step=1.0, label_visibility="collapsed",
                        help="Prix d'exercice de l'option", key="t1_K")
    st.session_state["shared_K"] = K

    mr = S / K if K > 0 else 1
    if S == K:                                                        pc, pt = "matm", "ATM"
    elif (otype == "call" and S > K) or (otype == "put" and S < K):  pc, pt = "mitm", "ITM"
    else:                                                             pc, pt = "motm", "OTM"
    st.markdown(f'<span class="mpill {pc}">{pt} \u00b7 {mr:.3f}</span>', unsafe_allow_html=True)

    # ── Maturité ────────────────────────────────────────────
    st.markdown('<div class="sb-title">Maturité</div>', unsafe_allow_html=True)
    _mc1, _mc2, _mc3 = st.columns(3)
    with _mc1:
        field_label("Années")
        _ty = st.number_input("A", 0, 30, 0, 1, key="p1_y",
                              label_visibility="collapsed", help="Années")
    with _mc2:
        field_label("Mois")
        _tm = st.number_input("M", 0, 11, 1, 1, key="p1_m",
                              label_visibility="collapsed", help="Mois")
    with _mc3:
        field_label("Jours")
        _td = st.number_input("J", 0, 30, 15, 1, key="p1_d",
                              label_visibility="collapsed", help="Jours")
    T = mat_from_ymd(_ty, _tm, _td)

    # ── Taux, Dividende, Volatilité ─────────────────────────
    st.markdown('<div class="sb-title">Paramètres de marché</div>', unsafe_allow_html=True)
    field_label("Taux sans risque  r  (%)")
    r = st.slider("r", 0.0, 10.0,
                  float(st.session_state.get("shared_r", 2.5)), 0.1,
                  label_visibility="collapsed",
                  help="Taux d'intérêt annuel sans risque (taux BCE \u2248 2,5% mi-2026)", key="t1_r") / 100
    st.session_state["shared_r"] = r * 100

    field_label("Dividende  q  (%)")
    q_div = st.slider("q", 0.0, 20.0,
                      float(st.session_state.get("shared_q", 0.0)), 0.1,
                      label_visibility="collapsed",
                      help="Dividende annuel continu. 0 si pas de dividende.", key="t1_q") / 100
    st.session_state["shared_q"] = q_div * 100

    field_label("Volatilité  \u03c3  (%)")
    sigma = st.slider("sg", 1.0, 150.0,
                      float(st.session_state.get("shared_sigma", 20.0)), 0.5,
                      label_visibility="collapsed",
                      help="Volatilité implicite annualisé.", key="t1_sigma") / 100
    st.session_state["shared_sigma"] = sigma * 100

    # ── Volatilité implicite ────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sb-title">Volatilité implicite</div>', unsafe_allow_html=True)
    field_label("Prix de marché observé (\u20ac)")
    mkt = st.number_input("Prix observé (\u20ac)", value=0.0, step=0.01, min_value=0.0,
                          label_visibility="collapsed",
                          help="Prix affiché chez votre courtier \u2192 retrouve la vol. implicite",
                          key="sb_mkt")
    if mkt > 0:
        iv = implied_vol(mkt, S, K, T, r, q_div, otype)
        if not np.isnan(iv):
            st.markdown(f'<div class="ivr">\u03c3 impl. = {iv*100:.3f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ivf">Prix hors bornes BS</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Pricer & Grecques", "Stratégies", " Stratégies Customs", "Glossaire"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — Zone principale pleine largeur
# ═══════════════════════════════════════════════════════════
with tab1:
    price_raw = bs_price(S,K,T,r,sigma,q_div,otype)
    G_raw = bs_greeks(S,K,T,r,sigma,q_div,otype)
    # Appliquer le signe de la position (Long = +1, Short = -1)
    price = price_raw * pos_sign
    G = {k: v * pos_sign for k, v in G_raw.items()}
    d1v=(np.log(S/K)+(r-q_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T)) if T>1e-9 and sigma>1e-9 else 0
    d2v=d1v-sigma*np.sqrt(T) if T>1e-9 else 0
    badge_class="ph-c" if otype=="call" else "ph-p"
    pos_label = "LONG" if pos_sign == 1 else "SHORT"
    badge_text = f"{pos_label} {'CALL' if otype=='call' else 'PUT'}"

    # ── Prix hero + Greeks côte à côte ──────────────────────
    hero_col, greeks_col = st.columns([5, 7], gap="large")
    with hero_col:
        st.markdown(f"""
        <div class="ph">
          <div>
            <div class="ph-ey">Prix Black-Scholes</div>
            <div class="ph-row"><span class="ph-val">{price:.4f}</span><span class="ph-val" style="margin-left:6px">\u20ac</span></div>
            <div class="ph-sub">
              <span>d\u2081 = {d1v:.4f}</span><span>d\u2082 = {d2v:.4f}</span>
              <span>N(d\u2081) = {norm.cdf(d1v):.4f}</span><span>N(d\u2082) = {norm.cdf(d2v):.4f}</span>
            </div>
          </div>
          <span class="ph-badge {badge_class}">{badge_text}</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("Formule de Black-Scholes"):
            st.markdown(r"""
<div style="font-size:.78rem;line-height:2.0;color:#d4d4d8;font-family:'DM Mono',monospace">

**Prix d'un Call :**
$$C = S \, e^{-qT} \, N(d_1) \;-\; K \, e^{-rT} \, N(d_2)$$

**Prix d'un Put :**
$$P = K \, e^{-rT} \, N(-d_2) \;-\; S \, e^{-qT} \, N(-d_1)$$

**avec :**
$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\,T}{\sigma\sqrt{T}} \qquad d_2 = d_1 - \sigma\sqrt{T}$$

| Paramètre | Description |
|-----------|-------------|
| S | Prix du sous-jacent (spot) |
| K | Prix d'exercice (strike) |
| T | Temps jusqu'à l'expiration (en années) |
| r | Taux sans risque annuel |
| q | Rendement du dividende continu |
| N() | Fonction de répartition de la loi normale |

</div>
""", unsafe_allow_html=True)

        section_header("Exposition - lecture des risques")
        interp_col1, interp_col2 = st.columns(2)
        with interp_col1:
            for gn in ["delta","gamma","theta"]: a,b,c_=interp(gn,G[gn]); signal_card(a,b,c_)
        with interp_col2:
            for gn in ["vega","rho"]: a,b,c_=interp(gn,G[gn]); signal_card(a,b,c_)
            a,b,c_=gamma_theta_msg(G["gamma"],G["theta"]); signal_card(a,b,c_)

    with greeks_col:
        section_header("Les Grecques - Sensibilités de l'option")
        gdata=[("\u0394","Delta",G["delta"],".4f","#22c55e","Sensibilité au spot (\u20ac/\u20ac)"),
               ("\u0393","Gamma",G["gamma"],".5f","#a78bfa","Convexité du Delta"),
               ("\u0398","Theta",G["theta"],"+.4f","#f59e0b","Effet du temps (\u20ac/jour)"),
               ("\u03bd","Vega", G["vega"], ".4f","#3b82f6","Effet de la volatilité (\u20ac/%)"),
               ("\u03c1","Rho",  G["rho"],  "+.4f","#ef4444","Effet des taux (\u20ac/%)"),
               ("\u039b","Vanna",G["vanna"],"+.4f","#71717a","Cross Delta / Volatilité")]
        cards_html = ''.join(f'<div>{greek_card_html(sym,nm,v,fmt,col_c,desc)}</div>' for sym,nm,v,fmt,col_c,desc in gdata)
        st.markdown(f'<div class="greeks-grid">{cards_html}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Tableau détaillé des grecques"):
            df_g=pd.DataFrame([
                {"Grec":"\u0394 Delta","Valeur":f"{G['delta']:+.6f}","Unité":"\u20ac/\u20ac","Sens":"+ si haussier"},
                {"Grec":"\u0393 Gamma","Valeur":f"{G['gamma']:.6f}","Unité":"-","Sens":"+ si convexité favorable"},
                {"Grec":"\u0398 Theta","Valeur":f"{G['theta']:+.6f}","Unité":"\u20ac/j","Sens":"+ si le temps aide"},
                {"Grec":"\u03bd Vega","Valeur":f"{G['vega']:.6f}","Unité":"\u20ac/%","Sens":"+ si vol. implicite \u2191"},
                {"Grec":"\u03c1 Rho","Valeur":f"{G['rho']:+.6f}","Unité":"\u20ac/%","Sens":"+ si taux \u2191"},
                {"Grec":"\u039b Vanna","Valeur":f"{G['vanna']:+.6f}","Unité":"-","Sens":"Cross delta/vol"},
            ])
            st.dataframe(df_g.set_index("Grec"),use_container_width=True)

    # ── Visualisations pleine largeur ────────────────────────
    section_header("Visualisations")
    svg1,svg2,svg3,svg4,svg5 = build_dashboard(S,K,T,r,sigma,q_div,otype,pos_sign)

    # Ligne 1
    chart_r1c1, chart_r1c2 = st.columns(2)
    with chart_r1c1: show_svg(svg1, full_width=True)
    with chart_r1c2: show_svg(svg2, full_width=True)

    with st.expander(" Comment lire ces graphiques"):
        st.markdown('<div class="chart-exp"><b>Prix de l\'option (gauche)</b> - La courbe orange montre comment le prix de option '
                    'évolue selon le spot. La ligne grise pointillé est la valeur intrinsèque (plancher). '
                    'L\'écart entre les deux est la valeur temps.<br>'
                    '<b>Delta (droite)</b> - Montre la pente de la courbe de prix. Un delta de 0.5 signifie que '
                    'votre option gagne ~0.50\u20ac pour chaque +1\u20ac du sous-jacent.</div>', unsafe_allow_html=True)

    # Ligne 2
    chart_r2c1, chart_r2c2 = st.columns(2)
    with chart_r2c1: show_svg(svg3, full_width=True)
    with chart_r2c2: show_svg(svg4, full_width=True)

    with st.expander("Comment lire ces graphiques"):
        st.markdown('<div class="chart-exp"><b>Gamma (gauche)</b> - Pic maximal au strike (ATM). '
                    'Plus le gamma est élevé, plus votre delta accélère vite - utile pour les acheteurs, '
                    'risqué pour les vendeurs.<br>'
                    '<b>Vega (droite)</b> - Montre comment le prix réagit à la volatilité. '
                    'La ligne verticale orange marque votre volatilité actuelle.</div>', unsafe_allow_html=True)

    # Ligne 3 : centre
    chart_r3_pad1, chart_r3c1, chart_r3_pad2 = st.columns([1,2,1])
    with chart_r3c1: show_svg(svg5, full_width=True)

    with st.expander("Comment lire ce graphique"):
        st.markdown('<div class="chart-exp"><b>Theta - Time Decay</b> - La courbe montre l\'érosion du prix '
                    'à mesure que le temps passe. La décroissance s\'accélère en approchant de l\'expiration '
                    '(courbe concave). La ligne verticale orange marque votre maturité actuelle.</div>', unsafe_allow_html=True)

    # ── Analyse de scénarios ──
    section_header("Analyse de scénarios - Stress Test")
    st.markdown('<div style="font-size:.78rem;color:var(--t2);line-height:1.7;padding:6px 0 10px">'
                'Matrice montrant le prix de votre option pour différentes '
                'variations simultanées du spot et de la volatilité implicite. '
                'La case centrale (ref) correspond à vos paramètres actuels.</div>', unsafe_allow_html=True)
    st.markdown(scenario_grid_html(S, K, T, r, sigma, q_div, otype, pos_sign), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 2
# ═══════════════════════════════════════════════════════════
with tab2:
    p1,p2,p3,p4=st.columns(4)
    with p1:
        S2=st.number_input("Spot S\u2080",value=float(st.session_state.get("shared_S",100.0)),step=1.0,key="s2",help="Prix actuel du sous-jacent")
    with p2:
        K2=st.number_input("Strike K central",value=float(st.session_state.get("shared_K",100.0)),step=1.0,key="k2",
                           help="Strike ATM. Les strikes OTM sont calculés à \u00b17% automatiquement")
    with p3:
        r2=st.number_input("Taux r (%)",value=float(st.session_state.get("shared_r",2.5)),step=0.1,key="r2",help="Taux sans risque (%)") / 100
    with p4:
        q2=st.number_input("Dividende q (%)",value=float(st.session_state.get("shared_q",0.0)),step=0.1,min_value=0.0,max_value=20.0,
                           key="q2",help="Rendement du dividende annuel continu (%)") / 100
    pm1,pm2=st.columns(2)
    with pm1:
        field_label("Maturité Calls  (A / M / J)"); Tc2=mat_inline("s2c",1,0,0)
        sig_c2=st.slider("Vol. Calls \u03c3 (%)",1.0,100.0,20.0,0.5,key="sc2",
                         help="Volatilité implicite utilisé pour pricer les calls") / 100
    with pm2:
        field_label("Maturité Puts  (A / M / J)"); Tp2=mat_inline("s2p",1,0,0)
        sig_p2=st.slider("Vol. Puts \u03c3 (%)",1.0,100.0,20.0,0.5,key="sp2",
                         help="Volatilité implicite utilisé pour pricer les puts") / 100

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    section_header("Choisir une stratégie prédéfinie")

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

    detail_col1, detail_col2, detail_col3 = st.columns([2, 1, 1])
    with detail_col1:
        st.markdown(f'<div class="card" style="border-left:3px solid {info["color"]};height:100%">'
                    f'<div style="font-size:.96rem;font-weight:700;color:{info["color"]};margin-bottom:7px">{strat}</div>'
                    f'<div style="font-size:.76rem;color:#d4d4d8;line-height:1.7;margin-bottom:10px">{info["desc"]}</div>'
                    f'{legs_html(info["legs"])}</div>', unsafe_allow_html=True)
    with detail_col2:
        st.markdown(f'<div class="card" style="height:100%"><div class="ct">Payoff</div>'
                    f'<div style="display:grid;grid-template-columns:auto 1fr;gap:4px 10px;'
                    f'font-size:.73rem;line-height:1.8;color:#d4d4d8;align-items:baseline">'
                    f'<span>\U0001f4c8 <b>Gain max</b></span><span>{info["max_gain"]}</span>'
                    f'<span>\U0001f4c9 <b>Perte max</b></span><span>{info["max_loss"]}</span>'
                    f'<span>\u2696\ufe0f <b>Seuil</b></span><span>{info["be"]}</span>'
                    f'</div></div>', unsafe_allow_html=True)
    with detail_col3:
        tips = []
        if "Long Gamma" in info["greeks"]:  tips.append("\u26a1 <b>Long Gamma</b> - profite des grands mouvements")
        if "Short Gamma" in info["greeks"]: tips.append("\U0001f4b0 <b>Short Gamma</b> - encaisse le temps")
        if "Long Vega" in info["greeks"]:   tips.append("\U0001f4c8 <b>Long Vega</b> - profite si la vol. monte")
        if "Short Vega" in info["greeks"]:  tips.append("\U0001f4c9 <b>Short Vega</b> - profite si la vol. reste basse")
        st.markdown(f'<div class="card" style="height:100%"><div class="ct">Greeks</div>'
                    f'<div style="font-size:.72rem;color:#d4d4d8;margin-bottom:8px">{info["greeks"]}</div>'
                    f'<div style="font-size:.70rem;color:#d4d4d8;line-height:1.9">{"<br>".join(tips)}</div></div>',
                    unsafe_allow_html=True)

    section_header("Profil de gain/perte")
    st.markdown(f'<div style="font-size:.72rem;color:#d4d4d8;margin-bottom:6px">'
                f'Calls T={fmt_mat(Tc2)} \u03c3={sig_c2*100:.1f}%  \u00b7  Puts T={fmt_mat(Tp2)} \u03c3={sig_p2*100:.1f}%</div>',
                unsafe_allow_html=True)
    T_pct_2 = st.slider("Temps écoulé (%)", 1, 100, 100, 1, key="t_pct_2",
                         help="Positionnez le curseur pour voir le P&L à différents moments avant l'expiration") / 100
    T_avg_2 = (Tc2 + Tp2) / 2
    T_rem_2 = T_avg_2 * (1 - T_pct_2)
    st.markdown(f'<div class="pre-exp-label">{T_pct_2*100:.0f}% du temps écoulé - {fmt_mat(T_rem_2)} restant avant expiration</div>',
                unsafe_allow_html=True)
    _payoff_svg, _payoff_legend, _pnl_exp_t2, _pnl_pre_t2 = build_payoff(strat,S2,K2,Tc2,Tp2,r2,sig_c2,sig_p2,q2,T_pct_2)
    show_svg(_payoff_svg, full_width=True)
    # External legend
    _leg_html_t2 = ''.join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin:3px 10px">'
        f'<span style="width:18px;height:3px;border-radius:2px;background:{it["color"]};'
        f'{"border-top:1px dashed " + it["color"] if it["dash"] else ""};display:inline-block"></span>'
        f'<span style="font-size:.72rem;color:#d4d4d8">{it["label"]}</span></span>'
        for it in _payoff_legend
    )
    st.markdown(
        f'<div style="text-align:center;padding:6px 0 14px;display:flex;flex-wrap:wrap;'
        f'justify-content:center;gap:2px 6px">{_leg_html_t2}</div>',
        unsafe_allow_html=True)

    # P&L stats
    _SR_t2 = np.linspace(S2*0.5, S2*1.5, 400)
    _mx_t2 = float(np.max(_pnl_exp_t2))
    _mn_t2 = float(np.min(_pnl_exp_t2))
    _sig_avg_t2 = (sig_c2 + sig_p2) / 2
    _T_avg_t2 = (Tc2 + Tp2) / 2
    _pp_t2 = probability_of_profit(_pnl_exp_t2, _SR_t2, S2, _T_avg_t2, r2, _sig_avg_t2, q2)

    _rr_col, _pp_col = st.columns(2)
    with _rr_col:
        st.markdown(risk_reward_html(_mx_t2, _mn_t2), unsafe_allow_html=True)
    with _pp_col:
        st.markdown(prob_bar_html(_pp_t2), unsafe_allow_html=True)

    section_header("Grecques indicatifs - leg ATM")
    G2c=bs_greeks(S2,K2,Tc2,r2,sig_c2,q2,"call"); G2p=bs_greeks(S2,K2,Tp2,r2,sig_p2,q2,"put")
    adj=-1 if strat in ["Short Straddle","Short Strangle","Short Butterfly"] else 1
    Gm={k:(G2c[k]+G2p[k])*adj/2 for k in G2c}
    _g2data=[("\u0394","Delta","delta",".4f","#22c55e","Sensibilité au prix"),
             ("\u0393","Gamma","gamma",".5f","#a78bfa","Convexité"),
             ("\u0398","Theta","theta","+.5f","#f59e0b","Effet du temps (\u20ac/j)"),
             ("\u03bd","Vega","vega",".4f","#3b82f6","Effet volatilité (\u20ac/%)")]
    _g2html = ''.join(f'<div>{greek_card_html(sym,nm,Gm[key],fmt,color,desc)}</div>' for sym,nm,key,fmt,color,desc in _g2data)
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">{_g2html}</div>', unsafe_allow_html=True)

    greek_interp1, greek_interp2 = st.columns(2)
    with greek_interp1:
        for gn in ["delta","gamma"]: a,b,c_=interp(gn,Gm[gn]); signal_card(a,b,c_)
    with greek_interp2:
        for gn in ["theta","vega"]: a,b,c_=interp(gn,Gm[gn]); signal_card(a,b,c_)
        a,b,c_=gamma_theta_msg(Gm["gamma"],Gm["theta"]); signal_card(a,b,c_)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.expander("Tableau comparatif - toutes les stratégies"):
        rows=[]
        for nm,inf in STRATEGIES.items():
            a=-1 if nm in ["Short Straddle","Short Strangle","Short Butterfly"] else 1
            Gi=bs_greeks(S2,K2,Tc2,r2,sig_c2,q2,"call")
            rows.append({"Stratégie":nm,"Delta":f"{Gi['delta']*a:+.4f}","Gamma":f"{Gi['gamma']*a:+.5f}",
                "Theta":f"{Gi['theta']*a:+.5f}","Vega":f"{Gi['vega']*a:.4f}",
                "Vue":inf["outlook"],"Gain max":inf["max_gain"],"Perte max":inf["max_loss"]})
        st.dataframe(pd.DataFrame(rows).set_index("Stratégie"),use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card" style="margin-bottom:14px;font-size:.77rem;color:#d4d4d8;line-height:1.7">'
                'Construisez librement une stratégie en combinant jusqu\'a <b>6 jambes</b> indépendantes.<br>'
                'Chaque jambe a ses propres paramètres. Le graphique final agrège tous les P&L à maturité.</div>',
                unsafe_allow_html=True)

    # Template selector
    tpl_names = ["- Personnalisé -"] + list(BUILDER_TEMPLATES.keys())
    tpl_sel = st.selectbox("\u25CF Charger un template", tpl_names, key="tpl_sel",
                           help="Sélectionnez une stratégie prédéfinie pour pré-remplir les jambes")
    if tpl_sel != "- Personnalisé -":
        tpl = BUILDER_TEMPLATES[tpl_sel]
        st.markdown(f'<div class="tpl-info">Ce template va configurer <b>{tpl["n"]} jambes</b> '
                    f'basés sur le strike K = {st.session_state.get("shared_K",100.0):.1f} \u20ac</div>',
                    unsafe_allow_html=True)
        if st.button("Appliquer le template", key="apply_tpl", use_container_width=True):
            K_base = st.session_state.get("shared_K", 100.0)
            S_base = st.session_state.get("shared_S", 100.0)
            sig_base = st.session_state.get("shared_sigma", 20.0)
            r_base = st.session_state.get("shared_r", 2.5)
            q_base = st.session_state.get("shared_q", 0.0)
            for i, leg in enumerate(tpl["legs"]):
                st.session_state[f"la_{i}"] = True
                st.session_state[f"ldir_state_{i}"] = leg["dir"]
                st.session_state[f"li_{i}"] = leg["inst"]
                st.session_state[f"ls_{i}"] = S_base
                st.session_state[f"lk_{i}"] = round(K_base * (1 + leg.get("K_pct", 0)), 1)
                st.session_state[f"lq_{i}"] = leg.get("qty", 1)
                st.session_state[f"lr_{i}"] = r_base
                st.session_state[f"lq_div_{i}"] = q_base
                st.session_state[f"lsig_{i}"] = sig_base
                st.session_state[f"ly_{i}"] = 1
                st.session_state[f"lm_{i}"] = 0
                st.session_state[f"ld_{i}"] = 0
            for i in range(len(tpl["legs"]), 6):
                st.session_state[f"la_{i}"] = False
            st.session_state["n_legs_slider"] = tpl["n"]
            st.session_state["_tpl_name"] = tpl_sel
            st.rerun()

    _tpl_default = st.session_state.pop("_tpl_name", "")
    sname=st.text_input("\u25CF Nom de la stratégie",value=_tpl_default,placeholder="Ma stratégie",
                        help="Donnez un nom à votre stratégie - affiché sur le graphique")
    if not sname: sname="Ma stratégie"
    n_legs=st.slider("Nombre de jambes",1,6,2,1,key="n_legs_slider",help="Chaque jambe est une option indépendante")
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

        # ── Separator between legs ──
        if i > 0:
            st.markdown(f'<div style="height:1px;margin:20px 0;'
                        f'background:linear-gradient(90deg,transparent,{lc2}33,transparent)"></div>',
                        unsafe_allow_html=True)

        # ── Activation checkbox ──
        st.markdown(f'<div style="font-size:1.05rem;font-weight:800;color:{lc2};margin:4px 0 6px;letter-spacing:-.3px">{ln}</div>',
                    unsafe_allow_html=True)
        active=st.checkbox("Activer",value=d["active"],key=f"la_{i}")

        if active:
            # ── Direction toggle ──
            direction=d["dir"]
            _dir_cols = st.columns([1,1,4])
            with _dir_cols[0]:
                if st.button("Achat", key=f"lbuy_{i}", use_container_width=True):
                    st.session_state[f"ldir_state_{i}"] = 1
                    st.rerun()
            with _dir_cols[1]:
                if st.button("Vente", key=f"lsell_{i}", use_container_width=True):
                    st.session_state[f"ldir_state_{i}"] = -1
                    st.rerun()
            direction = st.session_state.get(f"ldir_state_{i}", d["dir"])

            dir_cls = "buy" if direction==1 else "sell"
            dir_label = "ACHAT" if direction==1 else "VENTE"
            cls="lb" if direction==1 else "ls"

            # Card header
            st.markdown(
                f'<div class="lc {cls}" style="border-color:{lc2}">'
                f'<div class="lc-head lc-head-{dir_cls}">'
                f'<span class="lc-name" style="color:{lc2}">{ln}</span>'
                f'<span class="lc-dir-badge lc-dir-{dir_cls}">{dir_label}</span>'
                f'</div>', unsafe_allow_html=True)

            # Row 1: Instrument, Spot, Strike, Qty, Taux, Dividende
            lc1_,lc2_,lc3_,lc4_,lc5_,lc6_=st.columns(6)
            with lc1_:
                instrument=st.selectbox("Instrument",["call","put"],
                    index=0 if d["inst"]=="call" else 1,key=f"li_{i}",
                    help="Call : option d'achat \u00b7 Put : option de vente")
            with lc2_:
                S_l=st.number_input("Spot S\u2080 (\u20ac)",value=d["S"],step=1.0,key=f"ls_{i}",
                    help="Prix actuel du sous-jacent")
            with lc3_:
                K_l=st.number_input("Strike K (\u20ac)",value=d["K"],step=0.5,key=f"lk_{i}",
                    help="Prix d'exercice")
            with lc4_:
                qty=st.number_input("Quantité",1,100,d["qty"],1,key=f"lq_{i}",help="Nombre de contrats")
            with lc5_:
                r_l=st.number_input("Taux r (%)",value=d["r"]*100,step=0.1,key=f"lr_{i}",
                    help="Taux sans risque (%)") / 100
            with lc6_:
                q_l=st.number_input("Dividende q (%)",value=0.0,step=0.1,min_value=0.0,max_value=20.0,
                    key=f"lq_div_{i}",help="Rendement du dividende annuel continu") / 100

            # Row 2: Vol, Maturite (A/M/J)
            lc7_,lc8_,lc9_,lc10_=st.columns(4)
            with lc7_:
                sig_l=st.slider(f"Vol. \u03c3 (%)",1.0,150.0,d["sigma"]*100,0.5,key=f"lsig_{i}",
                    help="Volatilité implicite annualisé") / 100
            with lc8_:
                y_l=st.number_input("Maturité (Années)",0,30,d["y"],1,key=f"ly_{i}",help="Années")
            with lc9_:
                m_l=st.number_input("Maturité (Mois)",0,11,d["m"],1,key=f"lm_{i}",help="Mois")
            with lc10_:
                dj_l=st.number_input("Maturité (Jours)",0,30,d["d"],1,key=f"ld_{i}",help="Jours")

            # Computed values
            T_l=mat_from_ymd(y_l,m_l,dj_l)
            prem=bs_price(S_l,K_l,T_l,r_l,sig_l,q_l,instrument)

            # Prime box with gradient
            st.markdown(
                f'<div style="margin:4px 0 0;padding:12px 18px;border-radius:0 0 9px 9px;'
                f'background:linear-gradient(135deg,rgba(59,130,246,.08) 0%,rgba(124,58,237,.08) 100%);'
                f'border-top:1px solid rgba(124,58,237,.15);display:flex;align-items:center;justify-content:flex-end;gap:10px">'
                f'<span style="font-size:.72rem;font-weight:600;color:#d4d4d8">Prime unitaire</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:1.05rem;font-weight:700;'
                f'background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 100%,#c084fc 100%);'
                f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
                f'\u20ac{prem:.4f}</span>'
                f'</div></div>', unsafe_allow_html=True)

            legs_cfg.append(dict(active=True,dir=direction,inst=instrument,
                S=S_l,K=K_l,T=T_l,r=r_l,sigma=sig_l,q=q_l,qty=qty,
                label=f'{"Achat" if direction==1 else "Vente"} {instrument.upper()} K={K_l:.0f} \u00d7{qty}'))
        else:
            st.markdown(f'<div class="lc lo"><span class="ln" style="color:var(--t3)">{ln} - désactivé</span></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    active_legs=[l for l in legs_cfg if l["active"]]
    S_ref = active_legs[0]["S"] if active_legs else 100.0

    if not active_legs:
        st.info("Activez au moins une jambe pour afficher le graphique.")
    else:
        net_premium = sum(l["dir"]*bs_price(l["S"],l["K"],l["T"],l["r"],l["sigma"],l.get("q",0),l["inst"])*l["qty"] for l in active_legs)
        def portfolio_greek(k): return sum(l["dir"]*l["qty"]*bs_greeks(l["S"],l["K"],l["T"],l["r"],l["sigma"],l.get("q",0),l["inst"])[k] for l in active_legs)
        PD, PG, PT, PV = portfolio_greek("delta"), portfolio_greek("gamma"), portfolio_greek("theta"), portfolio_greek("vega")
        premium_label = "encaissé" if net_premium < 0 else "payé"
        premium_color = "#22c55e" if net_premium < 0 else "#ef4444"
        delta_color = "#22c55e" if PD > 0.05 else ("#ef4444" if PD < -0.05 else "#3b82f6")
        gamma_color = "#22c55e" if PG > 0 else "#ef4444"
        theta_color = "#22c55e" if PT > 0 else "#ef4444"
        vega_color  = "#22c55e" if PV > 0 else "#ef4444"

        st.markdown(f"""
        <div class="tbar">
          <div class="tbi"><div class="tbl">Prime nette</div>
            <div class="tbv" style="color:{premium_color}">\u20ac{abs(net_premium):.4f}</div>
            <div style="font-size:.68rem;color:#a1a1aa">{premium_label}</div></div>
          <div class="tbi"><div class="tbl">\u0394 Delta</div><div class="tbv" style="color:{delta_color}">{PD:+.4f}</div></div>
          <div class="tbi"><div class="tbl">\u0393 Gamma</div><div class="tbv" style="color:{gamma_color}">{PG:+.5f}</div></div>
          <div class="tbi"><div class="tbl">\u0398 Theta \u20ac/j</div><div class="tbv" style="color:{theta_color}">{PT:+.5f}</div></div>
          <div class="tbi"><div class="tbl">\u03bd Vega \u20ac/%</div><div class="tbv" style="color:{vega_color}">{PV:+.4f}</div></div>
        </div>""", unsafe_allow_html=True)

        section_header("Exposition du portefeuille")
        builder_interp1, builder_interp2 = st.columns(2)
        with builder_interp1:
            for gn,gv in [("delta",PD),("gamma",PG)]: a,b,c_=interp(gn,gv); signal_card(a,b,c_)
        with builder_interp2:
            for gn,gv in [("theta",PT),("vega",PV)]: a,b,c_=interp(gn,gv); signal_card(a,b,c_)
            a,b,c_=gamma_theta_msg(PG,PT); signal_card(a,b,c_)

        section_header("Graphique P&L")
        T_pct_3 = st.slider("Temps écoulé (%)", 1, 100, 100, 1, key="t_pct_3",
                             help="Positionnez le curseur pour voir le P&L avant l'expiration") / 100
        active_Ts_3 = [l["T"] for l in active_legs]
        T_avg_3 = sum(active_Ts_3)/len(active_Ts_3) if active_Ts_3 else 1.0
        T_rem_3 = T_avg_3 * (1 - T_pct_3)
        st.markdown(f'<div class="pre-exp-label">{T_pct_3*100:.0f}% du temps écoulé - {fmt_mat(T_rem_3)} restant</div>',
                     unsafe_allow_html=True)

        svg_c, total_pnl, _, legend_items, total_pre = build_custom_payoff(active_legs, S_ref, sname, T_pct_3)
        show_svg(svg_c, full_width=True)
        # Legende externe centree sous le graphique
        legend_html = ''.join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin:3px 10px">'
            f'<span style="width:18px;height:3px;border-radius:2px;background:{it["color"]};'
            f'{"border-top:1px dashed " + it["color"] if it["dash"] else ""};display:inline-block"></span>'
            f'<span style="font-size:.72rem;color:#d4d4d8">{it["label"]}</span></span>'
            for it in legend_items
        )
        st.markdown(
            f'<div style="text-align:center;padding:6px 0 14px;display:flex;flex-wrap:wrap;'
            f'justify-content:center;gap:2px 6px">{legend_html}</div>',
            unsafe_allow_html=True)

        # Risk/Reward + Probability
        mx_v=np.max(total_pnl); mn_v=np.min(total_pnl)
        _sig_avg_3 = np.mean([l["sigma"] for l in active_legs]) if active_legs else 0.2
        _SR_t3 = np.linspace(S_ref*0.5, S_ref*1.5, 400)
        _pp_3 = probability_of_profit(total_pnl, _SR_t3, S_ref, T_avg_3,
                                       active_legs[0]["r"] if active_legs else 0.05, _sig_avg_3,
                                       active_legs[0].get("q",0) if active_legs else 0)
        _rr3_col, _pp3_col = st.columns(2)
        with _rr3_col:
            st.markdown(risk_reward_html(float(mx_v), float(mn_v)), unsafe_allow_html=True)
        with _pp3_col:
            st.markdown(prob_bar_html(_pp_3), unsafe_allow_html=True)

        section_header("Profils de Greeks vs Prix")
        greek_svgs=build_custom_greeks(active_legs,S_ref)
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
            + ''.join(f'<div style="border-radius:10px;overflow:hidden">{svg}</div>' for svg in greek_svgs)
            + '</div>',
            unsafe_allow_html=True)

        be_n=len(np.where(np.diff(np.sign(total_pnl)))[0])
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
          <div class="tbi"><div class="tbl">Gain max borné</div><div class="tbv" style="color:#22c55e">\u20ac{mx_v:.2f}</div></div>
          <div class="tbi"><div class="tbl">Perte max borné</div><div class="tbv" style="color:#ef4444">\u20ac{mn_v:.2f}</div></div>
          <div class="tbi"><div class="tbl">Break-evens</div><div class="tbv" style="color:#f59e0b">{be_n}</div></div>
          <div class="tbi"><div class="tbl">Jambes actives</div><div class="tbv" style="color:#3b82f6">{len(active_legs)}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Détail par jambe"):
            rows_l=[]
            for i,leg in enumerate(active_legs):
                pr=bs_price(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],leg.get("q",0),leg["inst"])
                Gl=bs_greeks(leg["S"],leg["K"],leg["T"],leg["r"],leg["sigma"],leg.get("q",0),leg["inst"])
                rows_l.append({"Jambe":LNAMES[i],"Sens":"Achat" if leg["dir"]==1 else "Vente",
                    "Type":leg["inst"].upper(),"S\u2080":f"{leg['S']:.1f}","Strike":f"{leg['K']:.1f}",
                    "Maturité":fmt_mat(leg["T"]),"\u03c3%":f"{leg['sigma']*100:.1f}","Qté":leg["qty"],
                    "Prix":f"\u20ac{pr:.4f}","Co\u00fbt net":f'{"\u2212" if leg["dir"]==1 else "+"}\u20ac{pr*leg["qty"]:.4f}',
                    "\u0394":f"{leg['dir']*Gl['delta']*leg['qty']:+.4f}",
                    "\u0393":f"{leg['dir']*Gl['gamma']*leg['qty']:+.5f}",
                    "\u0398":f"{leg['dir']*Gl['theta']*leg['qty']:+.5f}",
                    "\u03bd":f"{leg['dir']*Gl['vega']*leg['qty']:+.4f}"})
            st.dataframe(pd.DataFrame(rows_l).set_index("Jambe"),use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 4 — Glossaire
# ═══════════════════════════════════════════════════════════
with tab4:

    def gls_card(sym, name, subtitle, body, color, rgb):
        return (f'<div class="gls-card" style="--gc:{color};--gc-rgb:{rgb}">'
                f'<div class="gls-head"><div class="gls-sym">{sym}</div>'
                f'<div class="gls-titles"><div class="gls-name">{name}</div>'
                f'<div class="gls-sub">{subtitle}</div></div></div>'
                f'<div class="gls-body">{body}</div></div>')

    st.markdown('<div style="margin-bottom:6px"><span style="font-size:1.1rem;font-weight:800;letter-spacing:-.3px;'
                'background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 35%,#c084fc 60%,#f0abfc 85%,#fafafa 100%);'
                '-webkit-background-clip:text;-webkit-text-fill-color:transparent">Glossaire</span>'
                '<span style="font-size:.74rem;color:var(--t3);margin-left:12px">'
                'Tous les concepts utilisés dans Options Lab</span></div>', unsafe_allow_html=True)

    # ── Section : Fondamentaux ──
    st.markdown('<div class="gls-divider">Fondamentaux</div>', unsafe_allow_html=True)
    cards = ""
    cards += gls_card("BS", "Black-Scholes", "Modèle de pricing",
        "Modèle mathématique de référence pour évaluer le prix théorique d'une <b>option europénne</b>. "
        "Hypothèses : volatilité constante, marchés sans friction, actif suivant un mouvement brownien géométrique. "
        "Formule fermé publié en 1973 par Fischer Black, Myron Scholes et Robert Merton.",
        "#3b82f6", "59,130,246")
    cards += gls_card("C", "Option Call", "Droit d'achat",
        "Contrat donnant le <b>droit d'acheter</b> un actif à un prix fixé (strike) avant une date donné. "
        "L'acheteur paie une prime pour ce droit. Gain potentiel <b>illimité</b> si l'actif monte, "
        "perte limité à la prime payé. Valeur intrinsèque = max(S \u2212 K, 0).",
        "#22c55e", "34,197,94")
    cards += gls_card("P", "Option Put", "Droit de vente",
        "Contrat donnant le <b>droit de vendre</b> un actif au strike. "
        "Sert de protection (assurance) contre une baisse ou d'outil de spéculation baissière. "
        "Gain max = K \u2212 prime. Valeur intrinsèque = max(K \u2212 S, 0).",
        "#a78bfa", "167,139,250")
    cards += gls_card("K", "Strike", "Prix d'exercice",
        "Prix d'exercice fixé dans le contrat d'option. C'est le <b>seuil</b> à partir duquel l'option "
        "a une valeur intrinsèque. Détermine si l'option est ITM, ATM ou OTM. "
        "Le choix du strike influence directement le co\u00fbt de la prime et le profil de risque.",
        "#f59e0b", "245,158,11")
    cards += gls_card("M", "Moneyness", "ITM / ATM / OTM",
        "Relation entre le spot et le strike. "
        "<b>ITM</b> (In The Money) : l'option a une valeur intrinsèque positive. "
        "<b>ATM</b> (At The Money) : spot \u2248 strike. "
        "<b>OTM</b> (Out of The Money) : pas de valeur intrinsèque. "
        "La moneyness détermine la composition du prix entre valeur intrinsèque et valeur temps.",
        "#06b6d4", "6,182,212")
    cards += gls_card("\u03c3", "Volatilité Implicite", "Anticipation du marché",
        "Volatilité future <b>anticipé par le marché</b>, extraite du prix observé des options via "
        "inversion du modèle BS (méthode de Brent). Indicateur clé : une IV élevé signifie que "
        "le marché anticipe de forts mouvements. Utilisé pour comparer les options entre elles "
        "indépendamment de leur prix absolu.",
        "#ef4444", "239,68,68")
    st.markdown(f'<div class="gls-grid">{cards}</div>', unsafe_allow_html=True)

    # ── Section : Les Grecques ──
    st.markdown('<div class="gls-divider">Les Grecques - Sensibilités</div>', unsafe_allow_html=True)
    cards2 = ""
    cards2 += gls_card("\u0394", "Delta", "Sensibilité au prix",
        "Variation du prix de l'option pour <b>1\u20ac de mouvement</b> du sous-jacent. "
        "Aussi interprété comme la <b>probabilité approximative</b> d'expirer ITM. "
        "Call : 0 à +1 \u00b7 Put : \u22121 à 0. "
        "Un delta de 0.50 signifie que l'option gagne ~0.50\u20ac si le sous-jacent monte de 1\u20ac.",
        "#22c55e", "34,197,94")
    cards2 += gls_card("\u0393", "Gamma", "Convexité",
        "Taux de variation du Delta : mesure la <b>convexité</b> de la position. "
        "Gamma élevé = le Delta change vite = la position est très sensible aux grands mouvements. "
        "Maximum quand l'option est ATM et proche de l'expiration. "
        "Les acheteurs d'options sont <b>long gamma</b> (les mouvements jouent en leur faveur).",
        "#a78bfa", "167,139,250")
    cards2 += gls_card("\u0398", "Theta", "Effet du temps",
        "Perte de valeur quotidienne due au passage du temps (<b>time decay</b>). "
        "Toujours négatif pour les acheteurs d'options - chaque jour qui passe érode la valeur temps. "
        "L'érosion s'accélère à l'approche de l'expiration (proportionnelle à 1/\u221aT). "
        "Contrepartie naturelle du Gamma : long gamma = short theta.",
        "#f59e0b", "245,158,11")
    cards2 += gls_card("\u03bd", "Vega", "Sensibilité à la volatilité",
        "Sensibilité du prix à une <b>variation de 1%</b> de la volatilité implicite. "
        "Les options longues ont un Vega positif : elles profitent si la vol monte. "
        "Maximum quand l'option est ATM et loin de l'expiration. "
        "Essentiel pour le trading de volatilité (acheter/vendre la vol plut\u00f4t que la direction).",
        "#3b82f6", "59,130,246")
    cards2 += gls_card("\u03c1", "Rho", "Sensibilité aux taux",
        "Variation du prix de l'option pour une <b>hausse de 1%</b> du taux sans risque. "
        "Impact généralement faible sauf pour les options à <b>longue maturité</b> "
        "ou en période de forte variation des taux directeurs (BCE, Fed). "
        "Positif pour les calls (hausse des taux favorable), négatif pour les puts.",
        "#ef4444", "239,68,68")
    cards2 += gls_card("\u039b", "Vanna", "Cross Delta/Vol",
        "Dérivé croisé : mesure comment le <b>Delta varie quand la volatilité bouge</b> "
        "(ou, de manière équivalente, comment le Vega varie quand le spot bouge). "
        "Crucial pour le <b>hedging dynamique</b> des books d'options : un Vanna élevé signifie que "
        "votre couverture en Delta se dérègle si la volatilité change. "
        "Particulièrement important pour les stratégies de trading de skew.",
        "#71717a", "113,113,122")
    st.markdown(f'<div class="gls-grid">{cards2}</div>', unsafe_allow_html=True)

    # ── Section : Concepts avancés ──
    st.markdown('<div class="gls-divider">Concepts avancés</div>', unsafe_allow_html=True)
    cards3 = ""
    cards3 += gls_card("P/L", "Payoff", "Profil de gain/perte",
        "Graphique montrant le <b>profit ou la perte</b> d'une position en fonction du prix du sous-jacent "
        "à l'expiration. Permet de visualiser les break-evens, le gain max et la perte max. "
        "Pour les stratégies multi-jambes, le payoff total est la somme des payoffs individuels.",
        "#22c55e", "34,197,94")
    cards3 += gls_card("BE", "Break-Even", "Seuil de rentabilité",
        "Prix du sous-jacent à l'expiration o\u00f9 la position ne génère <b>ni gain ni perte</b>. "
        "Pour un long call : BE = strike + prime payé. Pour un long put : BE = strike \u2212 prime. "
        "Les stratégies complexes peuvent avoir plusieurs break-evens.",
        "#f59e0b", "245,158,11")
    cards3 += gls_card("IV", "Smile de Volatilité", "Structure de marché",
        "En théorie BS, la volatilité est constante pour tous les strikes. En pratique, "
        "l'IV varie selon le strike et la maturité formant un <b>smile</b> (sourire) ou <b>skew</b>. "
        "Les puts OTM ont souvent une IV plus élevé (demande de protection), "
        "créant une asymétrie révélatrice du sentiment de marché.",
        "#a78bfa", "167,139,250")
    cards3 += gls_card("\u0393/\u0398", "Gamma-Theta", "Arbitrage fondamental",
        "Relation inverse centrale en trading d'options : <b>long gamma = short theta</b> et vice versa. "
        "Acheter des options (long gamma) donne un avantage sur les grands mouvements mais co\u00fbte du temps. "
        "Vendre des options (short gamma) encaisse le temps mais expose aux mouvements brusques. "
        "C'est le compromis fondamental de toute stratégie d'options.",
        "#06b6d4", "6,182,212")
    st.markdown(f'<div class="gls-grid">{cards3}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:32px 0 8px;font-size:.64rem;color:#3f3f46;letter-spacing:1px">OPTIONS LAB \u00b7 BLACK-SCHOLES</div>
<div style="text-align:center;padding:0 40px 24px;font-size:.64rem;color:#3f3f46;line-height:1.7;max-width:800px;margin:0 auto">
  Les résultats fournis par cette application reposent sur le modèle théorique de Black-Scholes
  et sont présentés à titre purement informatif et pédagogique. Ils ne constituent en aucun cas
  un conseil en investissement, une recommandation d'achat ou de vente, ni une garantie de résultat.
  Les marchés d'options comportent des risques significatifs, y compris la perte totale du capital investi.
  Les calculs peuvent contenir des approximations inhérentes au modèle (volatilité constante, marchés sans friction, etc.).
  Consultez un professionnel agréé avant toute décision d'investissement.
</div>
""", unsafe_allow_html=True)


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
  --t:#fafafa;--t2:#d4d4d8;--t3:#a1a1aa;--t4:#27272a;
  --rad:10px;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background:var(--bg)!important;color:var(--t);}
.main .block-container{padding:22px 30px 48px;max-width:1580px;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:var(--bg)!important;}
/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;
  padding-bottom:18px;border-bottom:1px solid var(--b1);margin-bottom:20px;}
.hdr-l{display:flex;align-items:center;gap:14px;}
.hdr-ico{width:52px;height:52px;border-radius:13px;
  background:linear-gradient(135deg,#3b82f6 0%,#7c3aed 40%,#c084fc 75%,#f472b6 100%);
  display:flex;align-items:center;justify-content:center;color:#fff;font-size:22px;font-weight:700;
  box-shadow:0 6px 20px rgba(124,58,237,.35),0 2px 8px rgba(59,130,246,.2);}
.hdr-t{font-size:1.55rem;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 35%,#c084fc 60%,#f0abfc 85%,#fafafa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hdr-s{font-size:.72rem;color:var(--t2);margin-top:3px;letter-spacing:.4px;}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--s1);border:1px solid var(--b1);border-radius:9px;padding:3px;gap:2px;}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif!important;font-size:.8rem;font-weight:600;
  color:var(--t3);border-radius:7px;padding:8px 22px;transition:all .2s;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--s3) 0%,rgba(59,130,246,.06) 100%)!important;
  box-shadow:0 1px 6px rgba(0,0,0,.4)!important;border:1px solid rgba(59,130,246,.15)!important;}
.stTabs [aria-selected="true"] p{background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 40%,#c084fc 70%,#f0abfc 100%)!important;
  -webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;font-weight:700!important;}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){color:var(--t2);}
/* Cards */
.card{background:var(--s1);border:1px solid var(--b1);border-radius:var(--rad);padding:18px;}
.ct{font-size:.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--t2);margin-bottom:12px;}
/* Price hero */
.ph{background:linear-gradient(135deg,var(--s1) 0%,rgba(59,130,246,.04) 100%);
  border:1px solid var(--b1);border-radius:12px;
  padding:22px 26px;display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:14px;
  position:relative;overflow:hidden;}
.ph::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(59,130,246,.3),transparent);}
.ph-ey{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--t);margin-bottom:8px;}
.ph-row{display:flex;align-items:baseline;gap:6px;}
.ph-cur{font-size:.95rem;color:var(--t3);font-family:'DM Mono',monospace;}
.ph-val{font-size:1.9rem;font-weight:600;letter-spacing:-1px;font-family:'DM Mono',monospace;line-height:1;
  background:linear-gradient(180deg,#f8fafc,#a1a1aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.ph-sub{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--t2);margin-top:10px;
  display:flex;gap:10px;flex-wrap:wrap;padding-top:8px;border-top:1px solid rgba(255,255,255,.04);}
.ph-badge{font-size:.72rem;font-weight:600;padding:4px 14px;border-radius:20px;align-self:flex-start;letter-spacing:.5px;}
.ph-c{background:rgba(34,197,94,.08);color:var(--g);border:1px solid rgba(34,197,94,.15);}
.ph-p{background:rgba(167,139,250,.08);color:var(--p);border:1px solid rgba(167,139,250,.15);}
/* Greek card */
.gc{background:var(--s2);border:1px solid var(--b1);border-radius:10px;padding:14px 12px;
  position:relative;overflow:hidden;transition:border-color .2s;height:100%;
  display:flex;flex-direction:column;}
.gc:hover{border-color:var(--acc,#3b82f6);}
.gc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--acc,#3b82f6),transparent);opacity:.6;}
.gc-sym{font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:700;color:var(--acc,#3b82f6);line-height:1;}
.gc-nm{font-size:.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--t2);margin-top:3px;}
.gc-v{font-family:'DM Mono',monospace;font-size:1.0rem;font-weight:700;padding-top:9px;}
.gc-d{font-size:.72rem;color:var(--t2);margin-top:5px;line-height:1.45;height:2.9em;overflow:hidden;}
/* Greek grid — responsive */
.greeks-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;}
@media(max-width:900px){.greeks-grid{grid-template-columns:repeat(3,1fr);}}
.greeks-grid>div{min-width:0;}
/* Signals */
.sig{display:flex;align-items:flex-start;gap:9px;padding:10px 13px;
  border-radius:8px;border:1px solid;margin:4px 0;font-size:.73rem;line-height:1.65;}
.dot{width:5px;height:5px;border-radius:100%;flex-shrink:0;margin-top:6px;}
.sg{background:rgba(34,197,94,.06);border-color:rgba(34,197,94,.2);color:#86efac;}
.sr{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.2);color:#fca5a5;}
.sb{background:rgba(59,130,246,.06);border-color:rgba(59,130,246,.2);color:#93c5fd;}
.so{background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.2);color:#fcd34d;}
.dg{background:#22c55e;}.dr{background:#ef4444;}.db{background:#3b82f6;}.do{background:#f59e0b;}
/* Sec header */
.sh{font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;
  color:var(--t);padding:16px 0 10px;border-bottom:1px solid var(--b1);margin-bottom:13px;}
/* Field label */
.fl{font-size:.74rem;font-weight:600;color:var(--t2);margin-bottom:4px;margin-top:10px;}
/* Mat badge */
.mb{font-family:'DM Mono',monospace;font-size:.73rem;color:var(--b);
  background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.18);
  border-radius:6px;padding:5px 12px;margin:5px 0 10px;text-align:center;display:block;}
/* Moneyness */
.mpill{font-size:.72rem;font-weight:600;padding:2px 8px;border-radius:4px;display:inline-block;}
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
.sc-nm{font-size:.86rem;font-weight:700;margin-bottom:5px;}
.sc-s{font-size:.70rem;color:var(--t2);line-height:1.6;}
.tag{display:inline-block;font-size:.68rem;font-weight:600;padding:2px 6px;border-radius:4px;margin:1px 2px;}
.tby{background:rgba(34,197,94,.12);color:#22c55e;}
.tse{background:rgba(239,68,68,.12);color:#ef4444;}
.tca{background:rgba(59,130,246,.12);color:#3b82f6;}
.tpu{background:rgba(167,139,250,.12);color:#a78bfa;}
/* Total bar */
.tbar{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0;}
.tbi{background:var(--s1);border:1px solid var(--b1);border-radius:8px;padding:12px 14px;text-align:center;}
.tbl{font-size:.72rem;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.8px;}
.tbv{font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:700;margin-top:4px;}
/* Leg card */
.lc{border-radius:10px;padding:0;margin-bottom:14px;border:1px solid;overflow:hidden;}
.lb{background:var(--s1);border-color:rgba(34,197,94,.22);}
.ls{background:var(--s1);border-color:rgba(239,68,68,.22);}
.lo{background:var(--s1);border-color:var(--b1);opacity:.5;padding:14px 18px;}
.lc-head{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;
  border-bottom:1px solid var(--b1);}
.lc-head-buy{background:linear-gradient(135deg,rgba(34,197,94,.06) 0%,rgba(34,197,94,.02) 100%);}
.lc-head-sell{background:linear-gradient(135deg,rgba(239,68,68,.06) 0%,rgba(239,68,68,.02) 100%);}
.lc-head-left{display:flex;align-items:center;gap:10px;}
.lc-name{font-size:.88rem;font-weight:800;}
.lc-dir-badge{font-size:.73rem;font-weight:700;padding:4px 12px;border-radius:6px;letter-spacing:.5px;}
.lc-dir-buy{background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);}
.lc-dir-sell{background:rgba(239,68,68,.1);color:#ef4444;border:1px solid rgba(239,68,68,.2);}
.ln{font-size:.86rem;font-weight:800;}
/* Leg direction toggle buttons */
.leg-toggle .stButton>button{padding:6px 12px!important;font-size:.72rem!important;}
/* Pinfo row */
.prow{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;}
.pi{flex:1;min-width:105px;background:var(--s2);border:1px solid var(--b1);border-radius:8px;padding:10px 12px;}
.pil{font-size:.72rem;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.8px;}
.piv{font-size:.80rem;font-weight:600;margin-top:3px;}
/* Inputs */
div[data-testid="stNumberInput"] input{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;font-family:'DM Mono',monospace!important;font-size:.83rem!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;}
div[data-testid="stTextInput"] input{background:var(--s2)!important;border:1px solid var(--b1)!important;
  border-radius:7px!important;color:var(--t)!important;}
.stSlider [data-baseweb="slider"]{padding:4px 0;}
[data-testid="stSlider"] label{color:var(--t)!important;font-size:.75rem!important;}
.stRadio>label{color:var(--t)!important;font-size:.75rem!important;}
.stRadio [data-testid="stMarkdownContainer"] p{font-size:.75rem!important;color:var(--t)!important;}
.stButton>button{background:linear-gradient(135deg,var(--s2) 0%,var(--s3) 100%)!important;
  border:1px solid var(--b2)!important;color:var(--t2)!important;
  font-family:'Inter',sans-serif!important;font-size:.76rem!important;font-weight:600!important;
  border-radius:8px!important;padding:8px 16px!important;width:100%!important;transition:all .2s!important;
  letter-spacing:.3px!important;}
.stButton>button:hover{border-color:#3b82f6!important;color:var(--t)!important;
  background:linear-gradient(135deg,rgba(59,130,246,.08) 0%,rgba(124,58,237,.06) 100%)!important;
  box-shadow:0 2px 8px rgba(59,130,246,.15)!important;}
.stButton>button:active{transform:scale(.98)!important;}
.stCheckbox [data-testid="stMarkdownContainer"] p{font-size:.75rem!important;color:var(--t)!important;}
.streamlit-expanderHeader{font-family:'Inter',sans-serif!important;font-size:.73rem!important;
  color:var(--t2)!important;background:var(--s1)!important;border:1px solid var(--b1)!important;border-radius:8px!important;}
.streamlit-expanderContent{background:var(--s1)!important;border:1px solid var(--b1)!important;border-top:none!important;}
label,.stMarkdown p{color:var(--t)!important;font-size:.78rem!important;}
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
section[data-testid="stSidebar"] .stMarkdown p{color:var(--t)!important;font-size:.76rem!important;}
section[data-testid="stSidebar"] [data-testid="stSlider"] label{color:var(--t2)!important;font-size:.73rem!important;}
section[data-testid="stSidebar"] hr{border-color:var(--b1);margin:10px 0;}
.sb-title{font-size:.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;
  background:linear-gradient(90deg,#60a5fa,#a78bfa,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  padding:10px 0 6px;border-bottom:1px solid var(--b1);margin-bottom:8px;}
.sb-badge{display:inline-flex;align-items:center;gap:6px;font-size:.73rem;font-weight:600;
  padding:4px 10px;border-radius:6px;margin-bottom:10px;}
.sb-Call{background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);}
.sb-Put{background:rgba(167,139,250,.1);color:#a78bfa;border:1px solid rgba(167,139,250,.2);}
/* Scenario grid */
.sc-grid{width:100%;border-collapse:separate;border-spacing:0;font-family:'DM Mono',monospace;font-size:.73rem;}
.sc-grid th{background:var(--s2);color:var(--t3);font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;padding:10px 8px;border:1px solid var(--b1);}
.sc-grid th:first-child{border-radius:8px 0 0 0;}.sc-grid th:last-child{border-radius:0 8px 0 0;}
.sc-grid td{padding:8px 6px;text-align:center;border:1px solid var(--b1);line-height:1.5;transition:background .15s;}
.sc-grid td:hover{background:rgba(59,130,246,.06);}
.sc-grid tr:last-child td:first-child{border-radius:0 0 0 8px;}
.sc-grid tr:last-child td:last-child{border-radius:0 0 8px 0;}
.sc-grid .sc-ref{background:rgba(59,130,246,.08);font-weight:700;}
.sc-grid .sc-pos{color:#22c55e;}.sc-grid .sc-neg{color:#ef4444;}.sc-grid .sc-zero{color:var(--t3);}
/* Risk/Reward + Probability equal height */
[data-testid="stHorizontalBlock"]:has(.rr){align-items:stretch;}
[data-testid="stHorizontalBlock"]:has(.rr) [data-testid="stColumn"]>div{height:100%;}
[data-testid="stHorizontalBlock"]:has(.rr) [data-testid="stColumn"]>div>div{height:100%;}
/* Risk/Reward card */
.rr{background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:16px 18px;margin:8px 0;height:100%;
  display:flex;flex-direction:column;justify-content:space-between;}
.rr-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:10px;}
.rr-label{font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.8px;}
.rr-ratio{font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:800;
  background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.rr-bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:8px;background:var(--s2);}
.rr-gain{background:linear-gradient(90deg,#22c55e,#16a34a);border-radius:4px 0 0 4px;transition:width .3s;}
.rr-loss{background:linear-gradient(90deg,#ef4444,#dc2626);border-radius:0 4px 4px 0;transition:width .3s;}
.rr-vals{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:.78rem;font-weight:600;}
/* Probability bar */
.prob{background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:16px 18px;margin:8px 0;height:100%;
  display:flex;flex-direction:column;justify-content:space-between;}
.prob-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px;}
.prob-label{font-size:.72rem;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.8px;}
.prob-val{font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:800;}
.prob-bar{height:8px;border-radius:4px;background:var(--s3);overflow:hidden;margin-bottom:8px;}
.prob-fill{height:100%;border-radius:3px;transition:width .3s;}
/* Chart explanation */
.chart-exp{font-size:.72rem;color:var(--t3);line-height:1.7;padding:4px 0;}
/* Template selector */
.tpl-info{font-size:.72rem;color:var(--t2);background:rgba(59,130,246,.05);border:1px solid rgba(59,130,246,.12);
  border-radius:8px;padding:10px 14px;margin:6px 0;line-height:1.6;}
/* Pre-expiration slider label */
.pre-exp-label{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--b);
  background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.12);
  border-radius:6px;padding:6px 12px;margin:6px 0;text-align:center;}
/* Glossary */
.gls-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:8px;}
.gls-card{background:var(--s1);border:1px solid var(--b1);border-radius:12px;padding:0;
  overflow:hidden;transition:border-color .2s,box-shadow .2s;display:flex;flex-direction:column;}
.gls-card:hover{border-color:var(--gc,#3b82f6);box-shadow:0 4px 20px rgba(0,0,0,.3);}
.gls-head{padding:16px 20px 12px;display:flex;align-items:center;gap:14px;
  border-bottom:1px solid var(--b1);
  background:linear-gradient(135deg,rgba(var(--gc-rgb,59,130,246),.06) 0%,transparent 100%);}
.gls-sym{font-family:'DM Mono',monospace;font-size:1.15rem;font-weight:800;color:var(--gc,#3b82f6);
  width:48px;height:48px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  background:rgba(var(--gc-rgb,59,130,246),.08);border:1px solid rgba(var(--gc-rgb,59,130,246),.15);flex-shrink:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 4px;}
.gls-titles{display:flex;flex-direction:column;gap:2px;}
.gls-name{font-size:.92rem;font-weight:800;color:var(--t);letter-spacing:-.2px;}
.gls-sub{font-size:.70rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:1px;}
.gls-body{padding:16px 20px 18px;font-size:.76rem;color:var(--t2);line-height:1.8;flex:1;}
.gls-body b{color:var(--t);font-weight:700;}
.gls-tag{display:inline-block;font-size:.64rem;font-weight:600;padding:2px 8px;border-radius:4px;
  margin-top:10px;background:rgba(var(--gc-rgb,59,130,246),.08);color:var(--gc,#3b82f6);
  border:1px solid rgba(var(--gc-rgb,59,130,246),.15);}
.gls-divider{margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--b1);
  font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;
  background:linear-gradient(90deg,#60a5fa,#a78bfa,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
</style>
""", unsafe_allow_html=True)
