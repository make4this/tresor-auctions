"""Éléments communs aux pages de l'application Streamlit (textes affichés en anglais)."""

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

# Couleurs : bleu marine d'Imoframe (proche du #1F4E79 du plan) et alerte du plan. Les
# graphiques à plusieurs séries reprennent la palette validée des figures du rapport (lisible
# par les daltoniens).
PRIMARY = "#172A45"
ALERT = "#C8553D"
SOFT_TEXT = "rgba(17, 17, 17, 0.55)"
CARD = "#F0F0EF"
SERIES = ("#2a78d6", "#eb6834")
MUTED = "#8a897f"
GRID = "#e4e3df"
CACHE_SECONDS = 3600
REPOSITORY_URL = f"https://github.com/{GITHUB_REPOSITORY}"
DISCLAIMER = "Academic project (emlyon business school, 2026). Not investment advice."

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
    """Mise en forme commune : nombres à l'anglaise, fond clair, grille discrète."""
    figure.update_layout(
        separators=".,",  # point décimal, virgule des milliers
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


# Polices et coins carrés d'Imoframe, en CSS : mis dans .streamlit/config.toml (font,
# headingFont, baseRadius...), ces réglages empêchaient l'application de démarrer sur Streamlit
# Community Cloud, alors qu'ils marchaient en local. Les icônes gardent leur propre police.
FONTS = (
    "https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600"
    "&family=Inter:wght@500;600&display=swap"
)
PAGE_CSS = f"""
<style>
@import url('{FONTS}');
p, li, label, td, th, input, textarea, summary, small, .stCaption,
[data-testid="stMarkdownContainer"],
[data-testid="stSidebarNavItems"] a > span:not([data-testid="stIconMaterial"]),
button p {{ font-family: "DM Sans", sans-serif !important; }}
h1, h2, h3, h4, [data-testid="stMetricValue"] {{ font-family: Inter, sans-serif !important;
  font-weight: 500 !important; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
button, input, textarea, [data-baseweb="select"] > div, [data-testid="stExpander"] details,
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"] {{ border-radius: 0 !important; }}
.tr-hero h1 {{ font-size: clamp(2.1rem, 5vw, 3.4rem); line-height: 1.05; letter-spacing: -0.045em;
  margin: 0 0 0.9rem 0; padding: 0; font-weight: 500; }}
.tr-hero h1 .tr-soft {{ color: rgba(17, 17, 17, 0.38); }}
.tr-hero p {{ font-size: 1.1rem; color: {SOFT_TEXT}; max-width: 46rem; margin: 0; }}
.tr-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: 0.75rem; margin: 0.5rem 0 0.5rem 0; }}
.tr-step {{ background: {CARD}; padding: 1.1rem 1.2rem; }}
.tr-step .tr-number {{ color: rgba(17, 17, 17, 0.32); font-size: 0.85rem; }}
.tr-step h4, .tr-card h4 {{ margin: 0.15rem 0 0.35rem 0; padding: 0; font-weight: 500;
  font-size: 1.2rem; letter-spacing: -0.01em; }}
.tr-step p, .tr-card p {{ margin: 0; color: {SOFT_TEXT}; font-size: 0.95rem; line-height: 1.45; }}
.tr-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr));
  gap: 0.75rem; margin: 0.5rem 0 0.5rem 0; }}
.tr-card {{ background: {CARD}; padding: 1rem 1.1rem 1.1rem 1.1rem; }}
.tr-card .tr-date {{ color: {SOFT_TEXT}; font-size: 0.85rem; }}
.tr-badge {{ display: inline-block; padding: 0.2rem 0.55rem; margin: 0.15rem 0 0.55rem 0;
  font-size: 0.85rem; font-weight: 600; }}
.tr-weak {{ background: {ALERT}; color: #FFFFFF; }}
.tr-strong {{ background: {PRIMARY}; color: #FFFFFF; }}
.tr-normal {{ background: #E4E3DF; color: #111111; }}
.tr-panel {{ background-color: {PRIMARY}; color: #FFFFFF; padding: 1.4rem 1.5rem;
  background-image: radial-gradient(120% 150% at 0% 0%, rgba(120, 150, 205, 0.42) 0%,
  rgba(60, 90, 150, 0.16) 38%, rgba(23, 42, 69, 0) 68%); }}
.tr-panel h4 {{ color: #FFFFFF; margin: 0 0 0.4rem 0; padding: 0; font-weight: 500; }}
.tr-panel p {{ color: rgba(255, 255, 255, 0.85); margin: 0 0 0.3rem 0; }}
.tr-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: 0.75rem; }}
.tr-fact {{ border-top: 1px solid #D9D8D3; padding-top: 0.8rem; }}
.tr-fact .tr-value {{ font-family: Inter, sans-serif; font-size: 2rem; font-weight: 500;
  letter-spacing: -0.03em; line-height: 1.1; }}
.tr-fact p {{ color: {SOFT_TEXT}; margin: 0.3rem 0 0 0; font-size: 0.95rem; }}
</style>
"""


def page_style() -> None:
    """Style d'Imoframe pour toutes les pages : polices, coins carrés, et classes des blocs de
    la page d'accueil (cartes, étapes, chiffres)."""
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def chart(figure: go.Figure, height: int = 380, container: object = st) -> None:
    """Affiche un graphique, sans la barre d'outils de Plotly (elle cache le titre sur
    téléphone ; le zoom à la souris et le survol restent disponibles)."""
    container.plotly_chart(  # type: ignore[attr-defined]
        style(figure, height), width="stretch", config={"displayModeBar": False}
    )


def date_en(value: pd.Timestamp, year: bool = True) -> str:
    """« Mon, Nov 16, 2026 », ou « Mon, Nov 16 » sans l'année."""
    text = f"{value:%a, %b} {value.day}"
    return f"{text}, {value.year}" if year else text


def day_en(text: object) -> str:
    """Date ISO (2026-11-12) en « Nov 12, 2026 » ; « – » si absente."""
    try:
        day = pd.Timestamp(str(text))
    except (ValueError, TypeError):
        return "–"
    return "–" if pd.isna(day) else f"{day:%b} {day.day}, {day.year}"


def moment_en(text: object) -> str:
    """Instant ISO avec fuseau (2026-11-14T23:17:00+00:00) en jour et heure de New York."""
    try:
        moment = pd.Timestamp(str(text)).tz_convert(NEW_YORK_TZ)
    except (ValueError, TypeError):  # illisible ou sans fuseau
        return "unknown"
    if pd.isna(moment):  # absent : None, NaN ou texte vide deviennent une date manquante
        return "unknown"
    return f"{date_en(moment)}, {moment:%I:%M %p} ET".replace(" 0", " ", 1)


def sidebar() -> None:
    """Fraîcheur des données et bouton de mise à jour, sur toutes les pages : court et sobre."""
    status = json_file(live.STATUS_FILE)
    quality = json_file("data_quality.json")
    fetch = json_file(FETCH_STATUS_FILE)
    with st.sidebar:
        st.caption(f"Updated {moment_en(status.get('run_at'))}")
        problems = []
        if fetch and ad.failed_sources(fetch):
            problems.append(f"{len(ad.failed_sources(fetch))} source(s) failed to update")
        if quality and quality.get("status") == "rouge":
            problems.append("data check failed")
        if problems:
            st.caption("⚠️ " + "; ".join(problems) + " (see Data quality)")
        help_text = (
            "Reloads the auction calendar and today's yields, then recomputes provisional"
            " forecasts with the existing models. Official forecasts are recorded each evening."
        )
        if st.button("Refresh now", width="stretch", help=help_text):
            _refresh()


def _refresh() -> None:
    from tresor import refresh

    with st.spinner("Downloading and computing…"):
        try:
            predictions, failure = refresh.light_refresh()
        except refresh.RefreshBusy:
            st.sidebar.info("A refresh is already running. Try again in a minute.")
            return
        except refresh.ModelsUnavailable:
            st.sidebar.warning("Models are not available right now. Please try again later.")
            return
        except Exception as error:  # l'application reste utilisable
            st.sidebar.error(f"Refresh failed: {error}")
            return
    st.session_state["provisional"] = predictions
    message = "Provisional forecast ready: see Upcoming auctions."
    if failure:
        st.sidebar.warning(f"Partial refresh ({failure}). {message}")
    else:
        st.sidebar.success(message)
