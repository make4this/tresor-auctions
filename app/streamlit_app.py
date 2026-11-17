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

from tresor.app_ui import page_style, top_bar  # noqa: E402

st.set_page_config(page_title="CoverCast · US Treasury auctions", page_icon="📈", layout="wide")

# Onglets en haut de page, comme sur un site : l'accueil d'abord, puis les pages détaillées.
PAGES = [
    st.Page("vues/accueil.py", title="Home", default=True),
    st.Page("vues/prochaines.py", title="Upcoming auctions"),
    st.Page("vues/suivi.py", title="Live track record"),
    st.Page("vues/backtest.py", title="Model performance"),
    st.Page("vues/explorer.py", title="History"),
    st.Page("vues/donnees.py", title="Dataset"),
    st.Page("vues/qualite.py", title="Data quality"),
    st.Page("vues/methodologie.py", title="Methodology"),
]

page = st.navigation(PAGES, position="top")
page_style()
top_bar()
page.run()
