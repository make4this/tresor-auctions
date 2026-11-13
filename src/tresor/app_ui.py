"""Éléments communs aux pages de l'application Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import live
from tresor.config import CLEAN_DIR, OUTPUT_DIR

# Couleurs du plan : principale et alerte. Les graphiques à plusieurs séries reprennent la
# palette validée des figures du rapport (lisible par les daltoniens).
PRIMARY = "#1F4E79"
ALERT = "#C8553D"
SERIES = ("#2a78d6", "#eb6834")
MUTED = "#8a897f"
GRID = "#e4e3df"
CACHE_SECONDS = 3600


# Lecture mise en cache une heure au plus (section 11 du plan). La date de modification du
# fichier fait partie de la clé : un fichier réécrit (passage de nuit, mise à jour) est relu
# tout de suite.


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_parquet(path: str, version: int) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_csv(path: str, version: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_json(path: str, version: int) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _version(path: Path) -> int:
    return path.stat().st_mtime_ns


def parquet(name: str, folder: str = "output") -> pd.DataFrame:
    """Table de data/output (ou data/clean)."""
    path = (OUTPUT_DIR if folder == "output" else CLEAN_DIR) / name
    return _read_parquet(str(path), _version(path))


def csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return _read_csv(str(path), _version(path))


def json_file(name: str) -> dict[str, object]:
    path = OUTPUT_DIR / name
    return _read_json(str(path), _version(path)) if path.exists() else {}


def today_new_york() -> pd.Timestamp:
    return live.new_york_today(pd.Timestamp.now(tz="UTC"))


def style(figure: go.Figure, height: int = 380) -> go.Figure:
    """Mise en forme commune : nombres à la française, fond clair, grille discrète."""
    figure.update_layout(
        separators=", ",  # virgule décimale, espace des milliers
        template="plotly_white",
        height=height,
        # Titre en haut, légende juste au-dessus du graphique : ils ne se chevauchent pas.
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        title={"x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 15}},
        font={"size": 13},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return figure


def date_fr(value: pd.Timestamp) -> str:
    days = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    return f"{days[value.day_of_week]} {value.day:02d}/{value.month:02d}/{value.year}"


def sidebar() -> None:
    """Fraîcheur des données et bouton de mise à jour, sur toutes les pages."""
    status = json_file(live.STATUS_FILE)
    quality = json_file("data_quality.json")
    with st.sidebar:
        st.markdown("**Données**")
        if status:
            run_at = pd.Timestamp(str(status["run_at"])).tz_convert("America/New_York")
            st.caption(f"Dernière prévision : {date_fr(run_at)} à {run_at:%H:%M} (New York)")
            market = status.get("market_last_dates", {})
            if isinstance(market, dict) and market.get("yield_10y"):
                last = pd.Timestamp(str(market["yield_10y"]))
                st.caption(f"Derniers taux de marché : {last:%d/%m/%Y}")
        if quality:
            label = {"vert": "bon", "orange": "à surveiller", "rouge": "problème"}
            st.caption(f"Contrôle qualité : {label.get(str(quality.get('status')), '?')}")
        if st.button("Mettre à jour maintenant", width="stretch"):
            _refresh()
        st.caption(
            "Recharge les sources légères et prévoit sans ré-entraîner. La prévision obtenue"
            " est provisoire : seul le passage de nuit l'inscrit au journal figé."
        )


def _refresh() -> None:
    from tresor.refresh import light_refresh

    with st.spinner("Téléchargement et calcul…"):
        try:
            predictions, failure = light_refresh()
        except FileNotFoundError:
            st.sidebar.warning("Modèles absents ici : la mise à jour se fait la nuit.")
            return
        except Exception as error:  # l'application reste utilisable
            st.sidebar.error(f"Mise à jour impossible : {error}")
            return
    st.session_state["provisional"] = predictions
    st.cache_data.clear()
    if failure:
        st.sidebar.warning(f"Mise à jour partielle. {failure}")
    else:
        st.sidebar.success("Données mises à jour.")
