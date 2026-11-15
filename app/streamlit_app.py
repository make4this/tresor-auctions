"""Application Streamlit : prévision des adjudications du Trésor américain (phase 9).

Lancement : `streamlit run app/streamlit_app.py` depuis la racine du dépôt.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Le code du projet est dans src/ : on le rend importable même sans installation du paquet
# (une seule fois : ce script est relancé à chaque interaction).
SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import streamlit as st  # noqa: E402
from fresh_project import load_fresh  # noqa: E402

# Après une mise à jour du dépôt, le code de src/ est relu s'il a changé (voir fresh_project).
load_fresh("tresor", SOURCE)

from tresor.app_ui import page_style, sidebar  # noqa: E402

st.set_page_config(page_title="US Treasury Auctions", page_icon="📈", layout="wide")

# L'accueil explique l'essentiel en mots simples ; les pages détaillées sont regroupées.
HOME = st.Page("vues/accueil.py", title="Home", icon=":material/home:", default=True)
DETAILS = [
    st.Page("vues/prochaines.py", title="Upcoming auctions", icon=":material/event:"),
    st.Page("vues/suivi.py", title="Live track record", icon=":material/fact_check:"),
    st.Page("vues/backtest.py", title="Model performance", icon=":material/query_stats:"),
    st.Page("vues/explorer.py", title="History", icon=":material/show_chart:"),
    st.Page("vues/qualite.py", title="Data quality", icon=":material/verified:"),
    st.Page("vues/methodologie.py", title="Methodology", icon=":material/menu_book:"),
]
PAGES = {"": [HOME], "More details": DETAILS}

page = st.navigation(PAGES)
page_style()
sidebar()
page.run()
