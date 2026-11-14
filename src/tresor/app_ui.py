"""Éléments communs aux pages de l'application Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import live
from tresor.config import CLEAN_DIR, GITHUB_REPOSITORY, NEW_YORK_TZ, OUTPUT_DIR
from tresor.sources.collect import FETCH_STATUS_FILE

# Couleurs du plan : principale et alerte. Les graphiques à plusieurs séries reprennent la
# palette validée des figures du rapport (lisible par les daltoniens).
PRIMARY = "#1F4E79"
ALERT = "#C8553D"
SERIES = ("#2a78d6", "#eb6834")
MUTED = "#8a897f"
GRID = "#e4e3df"
CACHE_SECONDS = 3600
REPOSITORY_URL = f"https://github.com/{GITHUB_REPOSITORY}"
DISCLAIMER = (
    "Projet étudiant (cours « Machine Learning for Finance », emlyon business school, 2026)."
    " Ces prévisions ne sont pas un conseil en investissement."
)

# Lecture mise en cache une heure au plus (section 11 du plan). La date de modification et la
# taille du fichier font partie de la clé : un fichier réécrit est relu tout de suite.


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_parquet(path: str, version: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_csv(path: str, version: tuple[int, int]) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _read_json(path: str, version: tuple[int, int]) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _version(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def parquet(name: str, folder: str = "output") -> pd.DataFrame:
    """Table de data/output (ou data/clean)."""
    path = (OUTPUT_DIR if folder == "output" else CLEAN_DIR) / name
    return _read_parquet(str(path), _version(path))


def csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return _read_csv(str(path), _version(path))


def json_file(name: str) -> dict[str, object]:
    """Fichier JSON de data/output, ou un dictionnaire vide s'il manque ou est illisible."""
    path = OUTPUT_DIR / name
    if not path.exists():
        return {}
    try:
        return _read_json(str(path), _version(path))
    except (OSError, ValueError):
        return {}


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


def chart(figure: go.Figure, height: int = 380, container: object = st) -> None:
    """Affiche un graphique, sans la barre d'outils de Plotly (elle cache le titre sur
    téléphone ; le zoom à la souris et le survol restent disponibles)."""
    container.plotly_chart(  # type: ignore[attr-defined]
        style(figure, height), width="stretch", config={"displayModeBar": False}
    )


def date_fr(value: pd.Timestamp, year: bool = True) -> str:
    days = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    text = f"{days[value.day_of_week]} {value.day:02d}/{value.month:02d}"
    return f"{text}/{value.year}" if year else text


def day_fr(text: object) -> str:
    """Date ISO (2026-11-12) en date française (12/11/2026) ; « – » si absente."""
    try:
        return f"{pd.Timestamp(str(text)):%d/%m/%Y}"
    except (ValueError, TypeError):
        return "–"


def moment_fr(text: object) -> str:
    """Instant ISO avec fuseau (2026-11-14T23:17:00+00:00) en jour et heure de New York."""
    try:
        moment = pd.Timestamp(str(text)).tz_convert(NEW_YORK_TZ)
    except (ValueError, TypeError):  # absent, illisible ou sans fuseau
        return "inconnue"
    return f"{date_fr(moment)} à {moment:%H:%M} (New York)"


def sidebar() -> None:
    """Fraîcheur des données et bouton de mise à jour, sur toutes les pages."""
    status = json_file(live.STATUS_FILE)
    quality = json_file("data_quality.json")
    fetch = json_file(FETCH_STATUS_FILE)
    with st.sidebar:
        st.markdown("**Données**")
        st.caption(f"Dernière prévision : {moment_fr(status.get('run_at'))}")
        market = status.get("market_last_dates", {})
        if isinstance(market, dict) and market.get("yield_10y"):
            st.caption(f"Derniers taux de marché : {day_fr(market['yield_10y'])}")
        if quality:
            label = {"vert": "bon", "orange": "à surveiller", "rouge": "problème"}
            st.caption(f"Contrôle qualité : {label.get(str(quality.get('status')), '?')}")
        if fetch:
            failed = ad.failed_sources(fetch)
            st.caption(
                f"⚠️ Dernier téléchargement : {len(failed)} source(s) en échec (page Qualité)"
                if failed
                else "Dernier téléchargement : toutes les sources reçues"
            )
        if st.button("Mettre à jour maintenant", width="stretch"):
            _refresh()
        st.caption(
            "Recharge les sources légères et prévoit sans réentraîner, sur une copie des"
            " données. La prévision obtenue est provisoire : seul le passage du soir l'inscrit"
            " au journal figé."
        )
        st.caption(f"{DISCLAIMER} [Code et journal du projet]({REPOSITORY_URL}).")


def _refresh() -> None:
    from tresor import refresh

    with st.spinner("Téléchargement et calcul…"):
        try:
            predictions, failure = refresh.light_refresh()
        except refresh.RefreshBusy:
            st.sidebar.info("Une mise à jour est déjà en cours : réessayer dans une minute.")
            return
        except refresh.ModelsUnavailable:
            st.sidebar.warning(
                "Modèles indisponibles ici pour le moment : la prévision du soir reste affichée,"
                " réessayer plus tard."
            )
            return
        except Exception as error:  # l'application reste utilisable
            st.sidebar.error(f"Mise à jour impossible : {error}")
            return
    st.session_state["provisional"] = predictions
    message = "Prévision provisoire calculée : voir la page « Prochaines enchères »."
    if failure:
        st.sidebar.warning(f"Mise à jour partielle ({failure}). {message}")
    else:
        st.sidebar.success(message)
