"""Algoritmos para resolver juegos estáticos de información completa en forma normal.

Un juego se representa con la clase ``Game``:
    - player_names: nombres de los jugadores
    - strategies:   lista (una por jugador) con los nombres de sus estrategias puras
    - payoffs:      lista (una por jugador) de arreglos numpy con forma (n_1, ..., n_N);
                    payoffs[i][a_1, ..., a_N] es el pago del jugador i en el perfil a.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Estructura del juego
# --------------------------------------------------------------------------- #
@dataclass
class Game:
    player_names: list
    strategies: list
    payoffs: list

    @property
    def n_players(self) -> int:
        return len(self.player_names)

    @property
    def shape(self) -> tuple:
        return tuple(len(s) for s in self.strategies)

    def profiles(self) -> list:
        return list(itertools.product(*[range(n) for n in self.shape]))

    def profile_label(self, prof) -> str:
        return "(" + ", ".join(self.strategies[i][a] for i, a in enumerate(prof)) + ")"

    def payoff_vector(self, prof) -> tuple:
        return tuple(float(self.payoffs[i][tuple(prof)]) for i in range(self.n_players))


def fmt_num(x: float) -> str:
    """Formatea un número como fracción simple si es posible (3/5) o decimal."""
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    f = Fraction(x).limit_denominator(100)
    if abs(float(f) - x) < 1e-9:
        return str(f)
    return f"{x:.4g}"


# --------------------------------------------------------------------------- #
# 1. Eliminación iterada de estrategias estrictamente dominadas
# --------------------------------------------------------------------------- #
@dataclass
class Elimination:
    round: int
    player: int
    strategy: int
    by: dict  # {estrategia_dominante: probabilidad}; una sola con prob 1 => dominación pura


def _restricted_matrix(game: Game, i: int, active: list) -> np.ndarray:
    """Pagos del jugador i: filas = sus estrategias activas, columnas = perfiles activos de los demás."""
    sub = game.payoffs[i][np.ix_(*active)]
    return np.moveaxis(sub, i, 0).reshape(sub.shape[i], -1)


def _find_dominator(M: np.ndarray, idx: int, allow_mixed: bool):
    """¿La fila idx de M está estrictamente dominada? Devuelve {fila: prob} o None."""
    others = [k for k in range(M.shape[0]) if k != idx]
    if not others:
        return None
    for k in others:  # dominación por una estrategia pura
        if np.all(M[k] > M[idx] + TOL):
            return {k: 1.0}
    if not allow_mixed or len(others) < 2:
        return None
    # LP: max t  s.t.  sum_k s_k M[k,c] - M[idx,c] >= t  para toda columna c
    m = len(others)
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-M[others].T, np.ones((M.shape[1], 1))])
    b_ub = -M[idx]
    A_eq = np.array([[1.0] * m + [0.0]])
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
        bounds=[(0, None)] * m + [(None, None)], method="highs",
    )
    if res.success and -res.fun > 1e-7:
        return {others[k]: float(w) for k, w in enumerate(res.x[:m]) if w > 1e-9}
    return None


def iesds(game: Game, allow_mixed: bool = True):
    """Devuelve (estrategias_activas, lista_de_eliminaciones)."""
    N = game.n_players
    active = [list(range(n)) for n in game.shape]
    steps: list = []
    rnd = 0
    while True:
        rnd += 1
        found = []
        for i in range(N):
            if len(active[i]) < 2:
                continue
            M = _restricted_matrix(game, i, active)
            for pos, s in enumerate(active[i]):
                dom = _find_dominator(M, pos, allow_mixed)
                if dom:
                    found.append((i, s, {active[i][k]: w for k, w in dom.items()}))
        if not found:
            break
        for i, s, by in found:
            steps.append(Elimination(rnd, i, s, by))
        for i, s, _ in found:
            active[i].remove(s)
    return active, steps


# --------------------------------------------------------------------------- #
# 2. Equilibrios de Nash en estrategias puras (N jugadores)
# --------------------------------------------------------------------------- #
def pure_nash(game: Game) -> list:
    eqs = []
    N = game.n_players
    for prof in game.profiles():
        ok = True
        for i in range(N):
            cur = game.payoffs[i][prof]
            for a in range(game.shape[i]):
                if a == prof[i]:
                    continue
                alt = prof[:i] + (a,) + prof[i + 1:]
                if game.payoffs[i][alt] > cur + TOL:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            eqs.append(prof)
    return eqs


# --------------------------------------------------------------------------- #
# 3. Equilibrios de Nash en estrategias mixtas (2 jugadores): enumeración de soportes
# --------------------------------------------------------------------------- #
def _solve_support(A, B, I, J):
    k = len(I)
    AI = A[np.ix_(I, J)]
    BI = B[np.ix_(I, J)]
    # y en J: A[I,J] y = v 1, sum y = 1
    M1 = np.zeros((k + 1, k + 1))
    M1[:k, :k] = AI
    M1[:k, k] = -1.0
    M1[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    # x en I: x^T B[I,J] = w 1, sum x = 1
    M2 = np.zeros((k + 1, k + 1))
    M2[:k, :k] = BI.T
    M2[:k, k] = -1.0
    M2[k, :k] = 1.0
    try:
        s1 = np.linalg.solve(M1, rhs)
        s2 = np.linalg.solve(M2, rhs)
    except np.linalg.LinAlgError:
        return None
    y, v = s1[:k], s1[k]
    x, w = s2[:k], s2[k]
    if np.any(y < -1e-9) or np.any(x < -1e-9):
        return None
    m, n = A.shape
    xf = np.zeros(m)
    yf = np.zeros(n)
    xf[list(I)] = np.clip(x, 0, None)
    yf[list(J)] = np.clip(y, 0, None)
    if np.any(A @ yf > v + 1e-8) or np.any(xf @ B > w + 1e-8):
        return None
    return xf, yf, float(v), float(w)


def mixed_nash_2p(game: Game) -> list:
    """Todos los equilibrios (puros y mixtos) de un juego de 2 jugadores con soportes de igual tamaño.

    Es exhaustivo en juegos no degenerados. En juegos degenerados (con empates) pueden
    existir continuos de equilibrios; aquí se reportan los puntos extremos que se detectan.
    """
    assert game.n_players == 2
    A, B = game.payoffs
    m, n = A.shape
    sols, seen = [], set()
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                s = _solve_support(A, B, I, J)
                if s is None:
                    continue
                x, y, v, w = s
                key = tuple(np.round(np.concatenate([x, y]), 6))
                if key in seen:
                    continue
                seen.add(key)
                sols.append({
                    "x": x, "y": y, "u1": v, "u2": w,
                    "pure": bool(np.max(x) > 1 - 1e-9 and np.max(y) > 1 - 1e-9),
                })
    return sols


# --------------------------------------------------------------------------- #
# 4. Equilibrio correlacionado (programación lineal, N jugadores)
# --------------------------------------------------------------------------- #
OBJECTIVES = {
    "utilitarian": "Maximizar la suma de pagos",
    "worst_utilitarian": "Minimizar la suma de pagos",
    "egalitarian": "Maximizar el pago del jugador que menos recibe",
    "player_max": "Maximizar el pago de un jugador",
    "player_min": "Minimizar el pago de un jugador",
}


@dataclass
class CEResult:
    p: np.ndarray        # distribución sobre perfiles, misma forma que el juego
    payoffs: list        # pago esperado de cada jugador
    objective: str


def correlated_equilibrium(game: Game, objective: str = "utilitarian", player: int = 0):
    N = game.n_players
    profs = game.profiles()
    nv = len(profs)
    index = {p: j for j, p in enumerate(profs)}
    U = np.array([[game.payoffs[i][p] for p in profs] for i in range(N)])

    rows = []
    for i in range(N):
        for a in range(game.shape[i]):
            for b in range(game.shape[i]):
                if a == b:
                    continue
                row = np.zeros(nv)
                for j, p in enumerate(profs):
                    if p[i] == a:
                        alt = p[:i] + (b,) + p[i + 1:]
                        row[j] = U[i, j] - U[i, index[alt]]
                rows.append(-row)  # sum p * (u(a)-u(b)) >= 0
    A_ub = np.array(rows)
    b_ub = np.zeros(len(rows))
    A_eq = np.ones((1, nv))

    if objective == "egalitarian":
        A_ub = np.hstack([A_ub, np.zeros((A_ub.shape[0], 1))])
        extra = np.hstack([-U, np.ones((N, 1))])  # z <= U_i . p
        A_ub = np.vstack([A_ub, extra])
        b_ub = np.concatenate([b_ub, np.zeros(N)])
        A_eq = np.hstack([A_eq, np.zeros((1, 1))])
        c = np.zeros(nv + 1)
        c[-1] = -1.0
        bounds = [(0, None)] * nv + [(None, None)]
    else:
        if objective == "utilitarian":
            c = -U.sum(axis=0)
        elif objective == "worst_utilitarian":
            c = U.sum(axis=0)
        elif objective == "player_max":
            c = -U[player]
        elif objective == "player_min":
            c = U[player]
        else:
            raise ValueError(objective)
        bounds = [(0, None)] * nv

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        return None
    p = np.clip(res.x[:nv], 0, None)
    p[p < 1e-10] = 0.0
    p = p / p.sum()
    pay = [float(U[i] @ p) for i in range(N)]
    return CEResult(p=p.reshape(game.shape), payoffs=pay, objective=objective)
