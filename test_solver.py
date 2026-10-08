"""Pruebas del solver con los juegos de ejemplo. Ejecutar: python tests/test_solver.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solver import Game, correlated_equilibrium, iesds, mixed_nash_2p, pure_nash  # noqa: E402


def g2(A, B, s1, s2):
    return Game(["J1", "J2"], [s1, s2], [np.array(A, float), np.array(B, float)])


def test_prisioneros():
    g = g2([[2, 0], [3, 1]], [[2, 3], [0, 1]], ["Callar", "Confesar"], ["Callar", "Confesar"])
    assert pure_nash(g) == [(1, 1)]
    sols = mixed_nash_2p(g)
    assert len(sols) == 1 and sols[0]["pure"]  # no hay equilibrio completamente mixto
    active, steps = iesds(g)
    assert active == [[1], [1]]


def test_problema2():
    g = g2([[6, 0, 4], [2, 4, 2], [0, 10, 2]], [[2, 6, 4], [12, 3, 5], [6, 0, 2]],
           ["Alto", "Medio", "Bajo"], ["Izq", "Centro", "Der"])
    active, steps = iesds(g, allow_mixed=True)
    assert active == [[0, 2], [0, 1]], active
    assert {(s.player, s.strategy) for s in steps} == {(0, 1), (1, 2)}
    active_pure, _ = iesds(g, allow_mixed=False)
    assert active_pure == [[0, 1, 2], [0, 1, 2]]
    assert pure_nash(g) == []
    sols = mixed_nash_2p(g)
    assert len(sols) == 1
    s = sols[0]
    assert np.allclose(s["x"], [3 / 5, 0, 2 / 5]) and np.allclose(s["y"], [5 / 8, 3 / 8, 0])
    assert np.isclose(s["u1"], 15 / 4) and np.isclose(s["u2"], 18 / 5)


def test_parque():
    g = g2([[20, 10], [30, 0]], [[20, 30], [10, 0]], ["F", "C"], ["F", "C"])
    assert sorted(pure_nash(g)) == [(0, 1), (1, 0)]
    sols = mixed_nash_2p(g)
    assert len(sols) == 3
    mixto = [s for s in sols if not s["pure"]][0]
    assert np.allclose(mixto["x"], [0.5, 0.5]) and np.allclose(mixto["u1"], 15)


def test_piedra_papel_tijera():
    A = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    g = g2(A, (-np.array(A)).tolist(), list("PAT"), list("PAT"))
    assert pure_nash(g) == []
    sols = mixed_nash_2p(g)
    assert len(sols) == 1
    assert np.allclose(sols[0]["x"], 1 / 3) and np.allclose(sols[0]["y"], 1 / 3)


def test_correlacionado_gallina():
    g = g2([[6, 2], [7, 0]], [[6, 7], [2, 0]], ["Virar", "Seguir"], ["Virar", "Seguir"])
    r = correlated_equilibrium(g, "utilitarian")
    # Aumann: 1/3 en (V,V), (V,S), (S,V) => pago esperado 5 + 1/3 = 15/3... suma 10.5
    assert np.isclose(sum(r.payoffs), 10.5), r.payoffs
    assert np.isclose(r.p.sum(), 1.0)
    # todo equilibrio de Nash es correlacionado: el pago utilitario CE >= el de cualquier Nash
    for s in mixed_nash_2p(g):
        assert sum(r.payoffs) >= s["u1"] + s["u2"] - 1e-9


def test_tres_jugadores():
    shape = (2, 2, 2)
    rng = np.random.default_rng(0)
    g = Game(list("ABC"), [["x", "y"]] * 3, [rng.integers(0, 9, shape).astype(float) for _ in range(3)])
    iesds(g)
    pure_nash(g)
    for obj in ("utilitarian", "worst_utilitarian", "egalitarian", "player_max", "player_min"):
        r = correlated_equilibrium(g, obj, 1)
        assert r is not None and np.isclose(r.p.sum(), 1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
