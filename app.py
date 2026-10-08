"""Solver de juegos estáticos de información completa (forma normal) con Streamlit."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
import streamlit as st

import plots
from solver import (
    Game, correlated_equilibrium, fmt_num, iesds, mixed_nash_2p, pure_nash, OBJECTIVES,
)

st.set_page_config(page_title="Solver de juegos estáticos", page_icon="🎲", layout="wide")

MAX_PROFILES = 1000


# --------------------------------------------------------------------------- #
# Ejemplos precargados
# --------------------------------------------------------------------------- #
def _bien_publico_3():
    shape = (2, 2, 2)
    pay = [np.zeros(shape) for _ in range(3)]
    for prof in itertools.product(range(2), repeat=3):
        contribuyen = sum(1 for a in prof if a == 0)
        for i in range(3):
            pay[i][prof] = (4 if contribuyen >= 2 else 0) - (3 if prof[i] == 0 else 0)
    return dict(names=["Ana", "Beto", "Carla"], strategies=[["Contribuir", "No contribuir"]] * 3,
                payoffs=[p.tolist() for p in pay])


PRESETS = {
    "Personalizado": None,
    "Dilema del prisionero (problema 1)": dict(
        names=["Prisionero 1", "Prisionero 2"],
        strategies=[["Callar", "Confesar"], ["Callar", "Confesar"]],
        payoffs=[[[2, 0], [3, 1]], [[2, 3], [0, 1]]]),
    "Eliminación + Nash mixto 3x3 (problema 2)": dict(
        names=["Jugador 1", "Jugador 2"],
        strategies=[["Alto", "Medio", "Bajo"], ["Izquierda", "Centro", "Derecha"]],
        payoffs=[[[6, 0, 4], [2, 4, 2], [0, 10, 2]], [[2, 6, 4], [12, 3, 5], [6, 0, 2]]]),
    "Parque de los vecinos (problema 3)": dict(
        names=["Vecino 1", "Vecino 2"],
        strategies=[["A favor", "En contra"], ["A favor", "En contra"]],
        payoffs=[[[20, 10], [30, 0]], [[20, 30], [10, 0]]]),
    "Piedra, papel o tijera (problema 4)": dict(
        names=["Cortés", "Cuauhtémoc"],
        strategies=[["Piedra", "Papel", "Tijera"], ["Piedra", "Papel", "Tijera"]],
        payoffs=[[[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]]),
    "Batalla de los sexos": dict(
        names=["Ella", "Él"],
        strategies=[["Ópera", "Fútbol"], ["Ópera", "Fútbol"]],
        payoffs=[[[2, 0], [0, 1]], [[1, 0], [0, 2]]]),
    "Gallina (ideal para equilibrio correlacionado)": dict(
        names=["Conductor 1", "Conductor 2"],
        strategies=[["Virar", "Seguir"], ["Virar", "Seguir"]],
        payoffs=[[[6, 2], [7, 0]], [[6, 7], [2, 0]]]),
    "Bien público con 3 jugadores": _bien_publico_3(),
}


# --------------------------------------------------------------------------- #
# Utilidades de interfaz
# --------------------------------------------------------------------------- #
def show(fig):
    st.plotly_chart(fig, width="stretch")


def mix_text(strategies, probs) -> str:
    parts = [f"{fmt_num(p)} {s}" for s, p in zip(strategies, probs) if p > 1e-9]
    return " + ".join(parts)


def default_tensor(P, i: int, shape: tuple) -> np.ndarray:
    if P and tuple(len(s) for s in P["strategies"]) == shape:
        return np.array(P["payoffs"][i], dtype=float)
    return np.zeros(shape)


def read_game() -> Game:
    """Lee el juego desde la barra lateral y la tabla de pagos."""
    with st.sidebar:
        st.header("1 · Define el juego")
        preset_name = st.selectbox("Ejemplo precargado", list(PRESETS))
        P = PRESETS[preset_name]
        n = int(st.number_input("Número de jugadores", min_value=2, max_value=5, step=1,
                                value=len(P["names"]) if P else 2, key=f"n_{preset_name}"))
        names, strats = [], []
        for i in range(n):
            dn = P["names"][i] if P and i < len(P["names"]) else f"Jugador {i + 1}"
            ds = ", ".join(P["strategies"][i]) if P and i < len(P["strategies"]) else "A, B"
            nm = st.text_input(f"Nombre del jugador {i + 1}", dn, key=f"name_{preset_name}_{i}").strip()
            raw = st.text_input(f"Estrategias de {nm or i + 1}", ds, key=f"strat_{preset_name}_{i}",
                                help="Sepáralas con comas. Ej.: Callar, Confesar")
            names.append(nm or f"Jugador {i + 1}")
            strats.append([s.strip() for s in raw.split(",") if s.strip()])

    if len(set(names)) != n:
        st.error("Los nombres de los jugadores deben ser distintos.")
        st.stop()
    for nm, ss in zip(names, strats):
        if len(ss) < 2:
            st.error(f"{nm} necesita al menos 2 estrategias.")
            st.stop()
        if len(set(ss)) != len(ss):
            st.error(f"{nm} tiene estrategias con nombres repetidos.")
            st.stop()
    shape = tuple(len(s) for s in strats)
    n_prof = int(np.prod(shape))
    if n_prof > MAX_PROFILES:
        st.error(f"El juego tiene {n_prof} perfiles; el máximo permitido es {MAX_PROFILES}.")
        st.stop()

    sig = hashlib.md5(repr((preset_name, names, strats)).encode()).hexdigest()[:10]
    st.subheader("Pagos")
    if n == 2:
        st.caption("Filas: estrategias del jugador 1 · Columnas: estrategias del jugador 2. "
                   "Haz doble clic en una celda para editarla.")
        cols = st.columns(2)
        payoffs = []
        for i in range(2):
            with cols[i]:
                st.markdown(f"**Pago de {names[i]}**")
                df = pd.DataFrame(default_tensor(P, i, shape), index=strats[0], columns=strats[1])
                ed = st.data_editor(df, key=f"ed_{sig}_{i}", width="stretch")
                payoffs.append(ed.to_numpy(dtype=float))
    else:
        st.caption("Una fila por perfil de estrategias. Edita las columnas de pagos.")
        profs = list(itertools.product(*[range(k) for k in shape]))
        data = {names[i]: [strats[i][p[i]] for p in profs] for i in range(n)}
        for i in range(n):
            tens = default_tensor(P, i, shape)
            data[f"Pago: {names[i]}"] = [float(tens[p]) for p in profs]
        df = pd.DataFrame(data)
        ed = st.data_editor(df, key=f"ed_{sig}", width="stretch", hide_index=True,
                            disabled=names)
        payoffs = []
        for i in range(n):
            arr = ed[f"Pago: {names[i]}"].to_numpy(dtype=float)
            payoffs.append(arr.reshape(shape))  # el orden de las filas es el de itertools.product

    if any(np.isnan(p).any() for p in payoffs):
        st.error("Hay celdas de pago vacías. Llénalas con números.")
        st.stop()
    return Game(names, strats, payoffs)


# --------------------------------------------------------------------------- #
# Pestañas
# --------------------------------------------------------------------------- #
def tab_iesds(game: Game):
    st.markdown("Se eliminan, ronda por ronda, las estrategias **estrictamente dominadas**. "
                "Una estrategia está estrictamente dominada si otra (pura o mixta) da un pago "
                "mayor contra *cualquier* combinación de las estrategias que aún sobreviven de los demás.")
    mixed = st.checkbox("Permitir dominación por estrategias mixtas", value=True)
    active, steps = iesds(game, allow_mixed=mixed)

    if not steps:
        st.info("Ninguna estrategia está estrictamente dominada: sobreviven todas.")
    else:
        st.subheader("Pasos")
        for r in sorted({s.round for s in steps}):
            st.markdown(f"**Ronda {r}**")
            for s in [x for x in steps if x.round == r]:
                dom_names = [game.strategies[s.player][k] for k in s.by]
                if len(s.by) == 1:
                    by = f"la estrategia pura «{dom_names[0]}»"
                else:
                    by = "la mezcla " + mix_text(dom_names, s.by.values())
                st.markdown(f"- {game.player_names[s.player]}: «{game.strategies[s.player][s.strategy]}» "
                            f"está estrictamente dominada por {by}.")

    st.subheader("Estrategias que sobreviven")
    for i in range(game.n_players):
        st.markdown(f"- **{game.player_names[i]}**: " + ", ".join(game.strategies[i][a] for a in active[i]))
    if all(len(a) == 1 for a in active):
        prof = tuple(a[0] for a in active)
        st.success(f"El juego es resoluble por dominancia: perfil {game.profile_label(prof)} con pagos "
                   f"{tuple(fmt_num(x) for x in game.payoff_vector(prof))}.")

    if game.n_players == 2:
        show(plots.matrix_figure(game, active=active,
                                 title="Tabla de pagos (en gris, lo eliminado)"))
    show(plots.elimination_figure(game, active, steps))


def tab_nash(game: Game):
    st.markdown("Un perfil es **equilibrio de Nash en estrategias puras** si ningún jugador puede "
                "ganar desviándose unilateralmente.")
    eqs = pure_nash(game)
    if game.n_players == 2:
        show(plots.matrix_figure(game, highlight=eqs, title="Equilibrios de Nash puros (bordes rojos)"))
    if not eqs:
        st.warning("No hay equilibrios de Nash en estrategias puras. Revisa la pestaña de estrategias mixtas.")
        return
    st.success(f"Se encontraron {len(eqs)} equilibrio(s) de Nash en estrategias puras.")
    rows = []
    for e in eqs:
        row = {"Perfil": game.profile_label(e)}
        for i in range(game.n_players):
            row[f"Pago {game.player_names[i]}"] = fmt_num(game.payoffs[i][e])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    show(plots.nash_payoffs_figure(game, eqs))


def tab_mixed(game: Game):
    if game.n_players != 2:
        st.info("El cálculo de equilibrios mixtos está implementado por ahora para juegos de **2 jugadores**.")
        return
    st.markdown("Se enumeran todas las parejas de soportes del mismo tamaño y se imponen las "
                "condiciones de **indiferencia**: cada jugador debe obtener el mismo pago esperado con "
                "todas las estrategias que usa, y no más con una que no use.")
    sols = mixed_nash_2p(game)
    if not sols:
        st.warning("No se encontró ningún equilibrio (puede deberse a un juego degenerado).")
        return
    rows = []
    for k, s in enumerate(sols, 1):
        rows.append({
            "#": k,
            "Tipo": "Puro" if s["pure"] else "Mixto",
            game.player_names[0]: mix_text(game.strategies[0], s["x"]),
            game.player_names[1]: mix_text(game.strategies[1], s["y"]),
            f"Pago {game.player_names[0]}": fmt_num(s["u1"]),
            f"Pago {game.player_names[1]}": fmt_num(s["u2"]),
        })
    n_mixed = sum(1 for s in sols if not s["pure"])
    st.success(f"{len(sols)} equilibrio(s) de Nash en total: {len(sols) - n_mixed} puro(s) y {n_mixed} mixto(s).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if game.shape == (2, 2):
        show(plots.best_response_figure(game, sols))
    show(plots.mixed_figure(game, sols[:8]))
    if len(sols) > 8:
        st.caption("Se grafican los primeros 8 equilibrios.")


def tab_correlated(game: Game):
    st.markdown("Un **equilibrio correlacionado** es una distribución de probabilidad sobre los perfiles, "
                "que un mediador sortea y recomienda en privado a cada jugador, tal que nadie quiere "
                "desobedecer su recomendación. Es un problema de programación lineal; "
                "hay muchos y se elige uno con un criterio.")
    keys = list(OBJECTIVES)
    c1, c2 = st.columns([2, 1])
    obj = c1.selectbox("Criterio de selección", keys, format_func=lambda k: OBJECTIVES[k])
    player = 0
    if obj in ("player_max", "player_min"):
        player = c2.selectbox("Jugador", range(game.n_players), format_func=lambda i: game.player_names[i])
    res = correlated_equilibrium(game, obj, player)
    if res is None:
        st.error("No se pudo resolver el problema lineal.")
        return
    st.subheader("Equilibrio seleccionado")
    show(plots.ce_distribution_figure(game, res.p))
    st.markdown("**Pagos esperados:** " + " · ".join(
        f"{game.player_names[i]} = {fmt_num(v)}" for i, v in enumerate(res.payoffs)))
    support = [(pr, res.p[pr]) for pr in game.profiles() if res.p[pr] > 1e-9]
    st.dataframe(pd.DataFrame({"Perfil": [game.profile_label(pr) for pr, _ in support],
                               "Probabilidad": [fmt_num(v) for _, v in support],
                               "Decimal": [round(float(v), 4) for _, v in support]}),
                 hide_index=True, width="stretch")

    st.subheader("Comparación de criterios")
    comp = {}
    for k in ("utilitarian", "worst_utilitarian", "egalitarian"):
        r = correlated_equilibrium(game, k)
        if r:
            comp[OBJECTIVES[k]] = r
    for i in range(game.n_players):
        r = correlated_equilibrium(game, "player_max", i)
        if r:
            comp[f"Máximo para {game.player_names[i]}"] = r
    show(plots.ce_comparison_figure(game, comp))


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
st.title("🎲 Solver de juegos estáticos de información completa")
st.caption("Ingresa un juego en forma normal y resuélvelo con cuatro métodos distintos.")

game = read_game()

t1, t2, t3, t4 = st.tabs([
    "1 · Eliminación iterada",
    "2 · Equilibrio de Nash",
    "3 · Estrategias mixtas",
    "4 · Equilibrio correlacionado",
])
with t1:
    tab_iesds(game)
with t2:
    tab_nash(game)
with t3:
    tab_mixed(game)
with t4:
    tab_correlated(game)
