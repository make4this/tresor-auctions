"""Application Streamlit : prévision des adjudications du Trésor américain (phase 9).

Lancement : `streamlit run app/streamlit_app.py` depuis la racine du dépôt.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Le code du projet est dans src/ : on le rend importable même sans installation du paquet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from tresor.app_ui import sidebar  # noqa: E402

st.set_page_config(page_title="Adjudications du Trésor", page_icon="📈", layout="wide")

PAGES = [
    st.Page(
        "vues/prochaines.py", title="Prochaines enchères", icon=":material/event:", default=True
    ),
    st.Page("vues/suivi.py", title="Suivi en direct", icon=":material/fact_check:"),
    st.Page("vues/backtest.py", title="Backtest", icon=":material/query_stats:"),
    st.Page("vues/explorer.py", title="Explorer", icon=":material/show_chart:"),
    st.Page("vues/qualite.py", title="Qualité des données", icon=":material/verified:"),
    st.Page("vues/methodologie.py", title="Méthodologie", icon=":material/menu_book:"),
]

page = st.navigation(PAGES)
sidebar()
page.run()
