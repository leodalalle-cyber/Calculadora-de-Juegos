# 🎲 Solver de juegos estáticos de información completa

App web en Streamlit para resolver juegos en forma normal con 2 a 5 jugadores.

| Pestaña | Qué hace | Gráfica |
|---|---|---|
| 1 · Eliminación iterada | Elimina estrategias estrictamente dominadas (puras y mixtas, vía programación lineal) | Tabla de pagos con lo eliminado en gris y ronda de eliminación |
| 2 · Equilibrio de Nash | Equilibrios en estrategias puras (N jugadores) | Tabla con equilibrios resaltados y pagos |
| 3 · Estrategias mixtas | Todos los equilibrios de Nash (puros y mixtos) por enumeración de soportes (2 jugadores) | Probabilidades de equilibrio y correspondencias de mejor respuesta (2x2) |
| 4 · Equilibrio correlacionado | Programación lineal con varios criterios de selección (N jugadores) | Distribución sobre perfiles y comparación de pagos |

## Uso local

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Publicarla desde GitHub (Streamlit Community Cloud, gratis)

1. Crea un repositorio en GitHub y sube estos archivos (`app.py`, `solver.py`, `plots.py`, `requirements.txt`, `README.md`).
2. Entra a <https://share.streamlit.io> e inicia sesión con tu cuenta de GitHub.
3. Elige **New app**, selecciona tu repositorio, la rama `main` y el archivo principal `app.py`.
4. Pulsa **Deploy**. Cada `git push` actualiza la app automáticamente.

## Estructura

- `app.py`: interfaz de Streamlit (entrada de datos y pestañas).
- `solver.py`: algoritmos (no depende de Streamlit, se puede usar por separado).
- `plots.py`: gráficas con Plotly.
- `tests/test_solver.py`: pruebas con los juegos de ejemplo (`python tests/test_solver.py`).

## Limitaciones conocidas

- Los equilibrios mixtos se calculan solo para 2 jugadores. Para 3 o más jugadores los sistemas de indiferencia son no lineales.
- La enumeración de soportes es exhaustiva en juegos no degenerados; en juegos con empates puede haber continuos de equilibrios y solo se reportan los extremos detectados.
- Máximo de 1000 perfiles de estrategias.
