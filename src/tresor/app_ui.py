"""Éléments communs aux pages de l'application Streamlit (textes affichés en anglais)."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import live
from tresor.config import CLEAN_DIR, GITHUB_REPOSITORY, NEW_YORK_TZ, OUTPUT_DIR
from tresor.sources.collect import FETCH_STATUS_FILE

# Couleurs du tableau de bord : barres foncées, panneaux blancs, hausse en vert, baisse en
# rouge. Les courbes reprennent la palette validée des figures du rapport (bleu, orange), plus
# un violet et un vert pour une 3e et une 4e maturité.
NAVY = "#0F172A"
PRIMARY = "#2563EB"
ALERT = "#DC2626"
UP = "#15803D"
SERIES = ("#2a78d6", "#eb6834", "#7C3AED", "#059669")
MUTED = "#8a897f"
GRID = "#e4e3df"
CACHE_SECONDS = 3600
REPOSITORY_URL = f"https://github.com/{GITHUB_REPOSITORY}"

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
        paper_bgcolor="rgba(0, 0, 0, 0)",  # le fond du panneau
        plot_bgcolor="#FFFFFF",
        height=height,
        # Titre en haut, légende juste au-dessus du graphique : ils ne se chevauchent pas.
        margin={"l": 10, "r": 10, "t": 90 if figure.layout.title.text else 40, "b": 10},
        title={"x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 15}},
        font={"size": 13},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return figure


# Style du tableau de bord, en CSS : mis dans .streamlit/config.toml (polices, coins), ces
# réglages empêchaient l'application de démarrer sur Streamlit Community Cloud. Les icônes
# gardent leur propre police.
FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
PAGE_CSS = f"""
<style>
@import url('{FONT}');
p, li, label, td, th, input, textarea, summary, small, .stCaption, h1, h2, h3, h4,
[data-testid="stMarkdownContainer"], button p,
[data-testid="stSidebarNavItems"] a > span:not([data-testid="stIconMaterial"]),
[data-testid="stNavSectionHeader"] {{ font-family: Inter, sans-serif !important; }}
h1, h2, h3, h4 {{ font-weight: 600 !important; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2.2rem; padding-left: 2rem; padding-right: 2rem; }}
[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #CBD5E1; }}
[data-testid="stSidebar"] [aria-current="page"] {{ background-color: #1E293B; }}
[data-testid="stSidebar"] [aria-current="page"] span {{ color: #FFFFFF; }}
[data-testid="stSidebar"] [data-testid="stNavSectionHeader"],
[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] * {{ color: #64748B; }}
[data-testid="stSidebar"] button {{ background-color: transparent; border-color: #334155; }}
[data-testid="stSidebar"] button:hover {{ border-color: {PRIMARY}; }}
.tr-topbar {{ background-color: {NAVY}; color: #FFFFFF; padding: 0.8rem 1.1rem;
  border-radius: 6px; display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.9rem; }}
.tr-topbar .tr-brand {{ font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }}
.tr-topbar .tr-meta {{ color: #94A3B8; font-size: 0.85rem; }}
.tr-intro {{ color: #4B5563; font-size: 0.95rem; margin: 0 0 0.9rem 0; }}
.tr-panel {{ background: #FFFFFF; border: 1px solid #E2E6EC; border-radius: 6px;
  margin-bottom: 0.9rem; overflow-x: auto; }}
.tr-panel-head {{ font-weight: 600; font-size: 1.05rem; padding: 0.7rem 0.9rem 0.5rem 0.9rem;
  color: #111827; }}
.tr-panel-note {{ color: #6B7280; font-size: 0.8rem; padding: 0.5rem 0.9rem 0.7rem 0.9rem; }}
table.tr-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem;
  font-variant-numeric: tabular-nums; }}
table.tr-table th {{ background: #F3F5F8; color: #374151; font-weight: 600; font-size: 0.8rem;
  padding: 0.45rem 0.75rem; border-top: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB;
  white-space: nowrap; }}
table.tr-table td {{ padding: 0.42rem 0.6rem; border-bottom: 1px solid #F1F3F6;
  white-space: nowrap; }}
table.tr-table td.tr-wrap {{ white-space: normal; }}
table.tr-table tr:last-child td {{ border-bottom: none; }}
.tr-right {{ text-align: right; }}
.tr-up {{ color: {UP}; }}
.tr-down {{ color: {ALERT}; }}
.tr-flat {{ color: #6B7280; }}
.tr-tag {{ display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
  font-size: 0.78rem; font-weight: 600; }}
.tr-weak {{ background: #FDECEC; color: #B91C1C; }}
.tr-strong {{ background: #E7F6EC; color: #166534; }}
.tr-normal {{ background: #F1F3F5; color: #374151; }}
.tr-grid {{ display: grid; grid-template-columns: 1.3fr 1fr 1fr; gap: 4px;
  padding: 0 0.9rem 0.2rem 0.9rem; font-size: 0.82rem; font-variant-numeric: tabular-nums; }}
.tr-grid .tr-cell {{ padding: 0.55rem 0.2rem; text-align: center; border-radius: 3px; }}
.tr-grid .tr-label {{ padding: 0.55rem 0.2rem; color: #374151; font-weight: 500; }}
.tr-grid .tr-colhead {{ text-align: center; color: #374151; font-weight: 600;
  padding: 0.2rem 0; }}
.tr-better {{ background: #DCF2E3; color: #14532D; }}
.tr-worse {{ background: #FBE3E3; color: #7F1D1D; }}
.tr-unclear {{ background: #F1F3F5; color: #374151; }}
.tr-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); }}
.tr-stats.tr-three {{ grid-template-columns: repeat(3, 1fr); }}
.tr-stats.tr-three .tr-stat {{ padding-right: 0.3rem; }}
.tr-stat {{ padding: 0.3rem 0.9rem 0.8rem 0.9rem; }}
.tr-stat .tr-value {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; }}
.tr-stat .tr-text {{ color: #6B7280; font-size: 0.82rem; line-height: 1.35; }}
.tr-dot {{ display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%;
  margin-right: 0.45rem; }}
[class*="st-key-panel"] {{ background: #FFFFFF; border: 1px solid #E2E6EC !important;
  border-radius: 6px; }}
@media (max-width: 1240px) {{
  .st-key-home_grid [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-home_grid [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100%; }}
}}
@media (max-width: 640px) {{
  table.tr-table td, table.tr-table th {{ padding: 0.4rem 0.5rem; }}
}}
</style>
"""


def page_style() -> None:
    """Style du tableau de bord pour toutes les pages."""
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def top_bar() -> None:
    """Bandeau foncé en haut de chaque page : nom du site et heure de la dernière mise à jour."""
    status = json_file(live.STATUS_FILE)
    st.markdown(
        f"""<div class="tr-topbar"><span class="tr-brand">US Treasury Auctions</span>
<span class="tr-meta">Data updated {escape(moment_en(status.get("run_at")))}</span></div>""",
        unsafe_allow_html=True,
    )


# --- Petits morceaux de HTML pour les panneaux (textes déjà échappés par l'appelant) --------


def panel(title: str, body: str, note: str = "") -> str:
    """Un panneau blanc : titre, contenu, note grise facultative."""
    foot = f'<div class="tr-panel-note">{note}</div>' if note else ""
    return (
        f'<div class="tr-panel"><div class="tr-panel-head">{escape(title)}</div>{body}{foot}</div>'
    )


def table(
    headers: list[str],
    rows: list[list[str]],
    right: set[int] | None = None,
    wrap: set[int] | None = None,
) -> str:
    """Tableau compact ; `right` : colonnes alignées à droite (nombres), `wrap` : colonnes dont
    le texte peut passer à la ligne (les autres restent sur une ligne)."""
    right, wrap = right or set(), wrap or set()

    def cell(tag: str, index: int, content: str) -> str:
        classes = " ".join(
            name for name, on in (("tr-right", index in right), ("tr-wrap", index in wrap)) if on
        )
        return (
            f'<{tag} class="{classes}">{content}</{tag}>'
            if classes
            else f"<{tag}>{content}</{tag}>"
        )

    head = "".join(cell("th", i, escape(h)) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows
    )
    return f'<table class="tr-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def signed(value: float, text: str) -> str:
    """Texte coloré selon le signe de la valeur : vert si elle monte, rouge si elle baisse."""
    css = "tr-up" if value > 0 else "tr-down" if value < 0 else "tr-flat"
    return f'<span class="{css}">{escape(text)}</span>'


def demand_tag(signal: str) -> str:
    """Étiquette colorée du signal de demande."""
    label, css = {ad.WEAK: ("Weaker", "tr-weak"), ad.STRONG: ("Stronger", "tr-strong"),
                  ad.NORMAL: ("Normal", "tr-normal")}.get(signal, ("n/a", "tr-normal"))  # fmt: skip
    return f'<span class="tr-tag {css}">{label}</span>'


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


def day_en(text: object, year: bool = True) -> str:
    """Date ISO (2026-11-12) en « Nov 12, 2026 » (ou « Nov 12 ») ; « – » si absente."""
    try:
        day = pd.Timestamp(str(text))
    except (ValueError, TypeError):
        return "–"
    if pd.isna(day):
        return "–"
    return f"{day:%b} {day.day}, {day.year}" if year else f"{day:%b} {day.day}"


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
    quality = json_file("data_quality.json")
    fetch = json_file(FETCH_STATUS_FILE)
    with st.sidebar:
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
