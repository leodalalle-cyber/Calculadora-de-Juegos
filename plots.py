"""Gráficas (Plotly) para la app de juegos."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver import Game, fmt_num

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
GREY = "#d4d4d8"


def _cell_text(game: Game, i: int, j: int) -> str:
    return f"{fmt_num(game.payoffs[0][i, j])} , {fmt_num(game.payoffs[1][i, j])}"


def matrix_figure(game: Game, active=None, highlight=None, title: str = "") -> go.Figure:
    """Tabla de pagos de un juego de 2 jugadores.

    active:    estrategias que sobreviven (las demás se pintan en gris).
    highlight: perfiles (i, j) a remarcar con borde rojo.
    """
    A, B = game.payoffs
    m, n = A.shape
    z = (A + B).astype(float)
    colorscale = "Blues"
    zmin = zmax = None
    if active is not None:
        mask = np.ones((m, n), dtype=bool)
        for i in range(m):
            for j in range(n):
                mask[i, j] = (i in active[0]) and (j in active[1])
        if not mask.all() and mask.any():
            lo, hi = z[mask].min(), z[mask].max()
            if hi - lo < 1e-9:
                hi = lo + 1.0
            elim = lo - (hi - lo)
            z = np.where(mask, z, elim)
            frac = (lo - elim) / (hi - elim)
            colorscale = [[0, GREY], [max(frac - 1e-6, 0), GREY], [frac, "#dbeafe"], [1, "#3b82f6"]]
            zmin, zmax = elim, hi
    text = [[_cell_text(game, i, j) for j in range(n)] for i in range(m)]
    fig = go.Figure(go.Heatmap(
        z=z, x=game.strategies[1], y=game.strategies[0], text=text, texttemplate="%{text}",
        textfont={"size": 16}, colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=False, xgap=4, ygap=4, hoverinfo="skip",
    ))
    for (i, j) in (highlight or []):
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="#dc2626", width=4))
    fig.update_xaxes(side="top", title=game.player_names[1])
    fig.update_yaxes(autorange="reversed", title=game.player_names[0])
    fig.update_layout(title=title, height=110 + 90 * m, margin=dict(l=10, r=10, t=90, b=10))
    return fig


def elimination_figure(game: Game, active, steps) -> go.Figure:
    """Ronda en la que se elimina cada estrategia (las supervivientes llegan hasta el final)."""
    last = max([s.round for s in steps], default=0) + 1
    removed = {(s.player, s.strategy): s.round for s in steps}
    labels, values, colors, texts = [], [], [], []
    for i in range(game.n_players):
        for a, name in enumerate(game.strategies[i]):
            labels.append(f"{game.player_names[i]} · {name}")
            if (i, a) in removed:
                values.append(removed[(i, a)])
                colors.append("#ef4444")
                texts.append(f"eliminada en la ronda {removed[(i, a)]}")
            else:
                values.append(last)
                colors.append("#22c55e")
                texts.append("sobrevive")
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           text=texts, textposition="inside", hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Ronda", dtick=1, range=[0, last + 0.2])
    fig.update_layout(height=60 + 34 * len(labels), margin=dict(l=10, r=10, t=30, b=10),
                      title="Destino de cada estrategia")
    return fig


def nash_payoffs_figure(game: Game, eqs) -> go.Figure:
    labels = [game.profile_label(e) for e in eqs]
    fig = go.Figure()
    for i in range(game.n_players):
        fig.add_bar(name=game.player_names[i], x=labels,
                    y=[game.payoffs[i][e] for e in eqs], marker_color=PALETTE[i % 5],
                    text=[fmt_num(game.payoffs[i][e]) for e in eqs], textposition="outside")
    fig.update_layout(barmode="group", title="Pagos en cada equilibrio de Nash puro",
                      yaxis_title="Pago", height=380, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def mixed_figure(game: Game, sols) -> go.Figure:
    k = len(sols)
    titles = []
    for idx, s in enumerate(sols, 1):
        kind = "puro" if s["pure"] else "mixto"
        for i in range(2):
            titles.append(f"Equilibrio {idx} ({kind}) · {game.player_names[i]}")
    fig = make_subplots(rows=k, cols=2, subplot_titles=titles, vertical_spacing=0.12 / max(k, 1) * 2)
    for r, s in enumerate(sols, 1):
        for c, key in enumerate(("x", "y"), 1):
            probs = s[key]
            fig.add_bar(x=game.strategies[c - 1], y=probs, marker_color=PALETTE[c - 1],
                        text=[fmt_num(p) for p in probs], textposition="outside",
                        showlegend=False, row=r, col=c)
            fig.update_yaxes(range=[0, 1.15], row=r, col=c)
    fig.update_layout(height=max(300, 260 * k), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def _br_path(d0: float, d1: float):
    """Mejor respuesta de un jugador con 2 estrategias.

    d(t) = d1 + (d0 - d1) t es la ganancia de jugar su estrategia 1 sobre la 2 cuando el
    rival juega su estrategia 1 con prob. t. Devuelve (t, prob. de jugar su estrategia 1).
    """
    slope = d0 - d1

    def val(t):
        d = d1 + slope * t
        return 1.0 if d > 1e-12 else (0.0 if d < -1e-12 else 0.5)

    ts = list(np.linspace(0, 1, 101))
    root = None
    if abs(slope) > 1e-12:
        r = -d1 / slope
        if 1e-9 < r < 1 - 1e-9:
            root = r
    if root is None:
        return ts, [val(t) for t in ts]
    left = [t for t in ts if t < root]
    right = [t for t in ts if t > root]
    T = left + [root, root] + right
    V = [val(t) for t in left] + [val(root - 1e-6), val(root + 1e-6)] + [val(t) for t in right]
    return T, V


def best_response_figure(game: Game, sols) -> go.Figure:
    """Correspondencias de mejor respuesta para juegos 2x2; los cruces son los equilibrios."""
    A, B = game.payoffs
    # J1 compara fila 0 vs fila 1 según la prob. q con que J2 juega columna 0
    d0 = A[0, 0] - A[1, 0]
    d1 = A[0, 1] - A[1, 1]
    q, p_br = _br_path(d0, d1)
    # J2 compara columna 0 vs columna 1 según la prob. p con que J1 juega fila 0
    e0 = B[0, 0] - B[0, 1]
    e1 = B[1, 0] - B[1, 1]
    p, q_br = _br_path(e0, e1)
    s1, s2 = game.strategies
    n1, n2 = game.player_names
    fig = go.Figure()
    fig.add_scatter(x=p_br, y=q, mode="lines", line=dict(color=PALETTE[0], width=4),
                    name=f"Mejor respuesta de {n1}")
    fig.add_scatter(x=p, y=q_br, mode="lines", line=dict(color=PALETTE[1], width=4),
                    name=f"Mejor respuesta de {n2}")
    fig.add_scatter(x=[s["x"][0] for s in sols], y=[s["y"][0] for s in sols], mode="markers+text",
                    marker=dict(size=14, color="#111827", line=dict(color="white", width=2)),
                    text=[f"({fmt_num(s['x'][0])}, {fmt_num(s['y'][0])})" for s in sols],
                    textposition="top center", name="Equilibrios de Nash")
    fig.update_xaxes(title=f"Prob. de que {n1} juegue «{s1[0]}»", range=[-0.05, 1.05])
    fig.update_yaxes(title=f"Prob. de que {n2} juegue «{s2[0]}»", range=[-0.05, 1.1])
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10),
                      title="Correspondencias de mejor respuesta",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def ce_distribution_figure(game: Game, p: np.ndarray) -> go.Figure:
    if game.n_players == 2:
        text = [[f"{100 * v:.1f}%" if v > 1e-9 else "" for v in row] for row in p]
        fig = go.Figure(go.Heatmap(
            z=p, x=game.strategies[1], y=game.strategies[0], text=text, texttemplate="%{text}",
            textfont={"size": 16}, colorscale="Purples", zmin=0, zmax=max(p.max(), 1e-9),
            showscale=False, xgap=4, ygap=4, hoverinfo="skip"))
        fig.update_xaxes(side="top", title=game.player_names[1])
        fig.update_yaxes(autorange="reversed", title=game.player_names[0])
        fig.update_layout(height=110 + 90 * p.shape[0], margin=dict(l=10, r=10, t=90, b=10),
                          title="Distribución de probabilidad sobre los perfiles")
        return fig
    profs = [pr for pr in game.profiles() if p[pr] > 1e-9]
    fig = go.Figure(go.Bar(x=[game.profile_label(pr) for pr in profs], y=[p[pr] for pr in profs],
                           marker_color="#7c3aed",
                           text=[f"{100 * p[pr]:.1f}%" for pr in profs], textposition="outside"))
    fig.update_layout(title="Perfiles con probabilidad positiva", yaxis_title="Probabilidad",
                      height=380, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def ce_comparison_figure(game: Game, results: dict) -> go.Figure:
    """Pago esperado de cada jugador bajo distintos criterios de selección del equilibrio correlacionado."""
    fig = go.Figure()
    labels = list(results.keys())
    for i in range(game.n_players):
        vals = [results[k].payoffs[i] for k in labels]
        fig.add_bar(name=game.player_names[i], x=labels, y=vals, marker_color=PALETTE[i % 5],
                    text=[fmt_num(v) for v in vals], textposition="outside")
    fig.update_layout(barmode="group", title="Pagos esperados según el criterio de selección",
                      yaxis_title="Pago esperado", height=400, margin=dict(l=10, r=10, t=60, b=10))
    return fig
