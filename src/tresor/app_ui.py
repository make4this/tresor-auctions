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
from tresor.config import CLEAN_DIR, GITHUB_REPOSITORY, NEW_YORK_TZ, OUTPUT_DIR, SIGNAL_THRESHOLD
from tresor.formats import en
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


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _public_csv(path: str, version: tuple[int, int]) -> bytes:
    return ad.public_table(pd.read_parquet(path)).to_csv(index=False).encode("utf-8")


def public_csv(name: str, folder: str = "output") -> bytes:
    """Fichier CSV téléchargeable d'une table, sans les colonnes à diffusion restreinte ; relu
    seulement quand le fichier change."""
    path = (OUTPUT_DIR if folder == "output" else CLEAN_DIR) / name
    return _public_csv(str(path), _version(path))


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
[data-testid="stNavSectionHeader"] {{ font-family: Inter, sans-serif !important; }}
h1, h2, h3, h4 {{ font-weight: 600 !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.85rem !important; padding: 0.2rem 0 0.3rem 0 !important; }}
.block-container {{ padding-top: 2.4rem; padding-left: 2rem; padding-right: 2rem; }}
/* L'en-tête de Streamlit ne garde que son menu, sans fond : l'en-tête du site est en dessous. */
[data-testid="stHeader"] {{ background: transparent; height: 2.2rem; }}
/* En-tête du site : nom, mise à jour, alerte, bouton, puis les onglets. */
.st-key-topbar {{ background-color: {NAVY}; border-radius: 8px; padding: 0.6rem 1rem 0.2rem 1rem;
  margin-bottom: 0.9rem; gap: 0.35rem; }}
.st-key-topnav {{ border-top: 1px solid #1E293B; padding-top: 0.15rem; flex-wrap: wrap;
  gap: 0.1rem 0.4rem !important; }}
.st-key-topnav [data-testid="stPageLink"] a {{ padding: 0.35rem 0.6rem; border-radius: 0;
  border-bottom: 2px solid transparent; background: transparent !important; }}
.st-key-topnav [data-testid="stPageLink"] a p,
.st-key-topnav [data-testid="stPageLink"] a span {{ color: #CBD5E1 !important;
  font-family: Inter, sans-serif !important; font-size: 0.9rem; font-weight: 500; }}
.st-key-topnav [data-testid="stPageLink"] a:hover p {{ color: #FFFFFF !important; }}
.st-key-topnav [class*="st-key-nav_on_"] a {{ border-bottom-color: #60A5FA; }}
.st-key-topnav [class*="st-key-nav_on_"] a p {{ color: #FFFFFF !important; font-weight: 600; }}
.st-key-topnav [class*="st-key-nav_"],
.st-key-topnav > [data-testid="stLayoutWrapper"] {{ width: auto !important;
  flex: 0 0 auto !important; min-width: 0 !important; }}
.st-key-topbar button {{ background: transparent; border-color: #334155; }}
.st-key-topbar button p, .st-key-topbar button span {{ color: #E2E8F0 !important; }}
.st-key-topbar button:hover {{ border-color: {PRIMARY}; }}
.tr-topline {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 1rem; }}
.tr-topline .tr-logo {{ margin-right: -0.5rem; flex-shrink: 0; }}
.tr-topline .tr-brand {{ color: #FFFFFF; font-weight: 700; font-size: 1.3rem;
  letter-spacing: -0.03em; }}
.tr-topline .tr-cast {{ color: #60A5FA; }}
.tr-topline .tr-tagline {{ color: #CBD5E1; font-size: 0.9rem; }}
.tr-topline .tr-meta {{ color: #94A3B8; font-size: 0.85rem; }}
.tr-topline .tr-alert {{ color: #FBBF24; font-size: 0.85rem; text-decoration: none; }}
.tr-topline .tr-alert:hover {{ text-decoration: underline; }}
/* Bandeau animé des enchères à venir : défile seul, s'arrête au survol. */
.tr-ticker {{ overflow: hidden; background: #FFFFFF; border: 1px solid #E2E6EC;
  border-radius: 6px; margin-bottom: 0.9rem; white-space: nowrap; }}
.tr-track {{ display: inline-flex; animation: tr-scroll var(--tr-duration, 60s) linear infinite; }}
.tr-ticker:hover .tr-track {{ animation-play-state: paused; }}
.tr-tick {{ display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.55rem 1.1rem;
  border-right: 1px solid #EEF1F5; font-size: 0.85rem; color: #374151;
  font-variant-numeric: tabular-nums; }}
.tr-tick b {{ color: #111827; }}
@keyframes tr-scroll {{ from {{ transform: translateX(0); }}
  to {{ transform: translateX(-50%); }} }}
/* Carte « Next auction » : glisse à l'arrivée. Seule la position bouge, jamais l'opacité :
   si le navigateur met l'animation en pause, la carte reste visible. */
.tr-hero {{ background: {NAVY}; color: #FFFFFF; border-radius: 6px; padding: 1.1rem 1.2rem;
  margin-bottom: 0.9rem; animation: tr-rise 0.6s ease-out; }}
.tr-hero .tr-kicker {{ color: #94A3B8; font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; }}
.tr-hero .tr-when {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
  margin: 0.2rem 0 0 0; line-height: 1.15; }}
.tr-hero .tr-what {{ color: #CBD5E1; font-size: 1rem; margin-bottom: 0.7rem; }}
.tr-hero .tr-tag {{ font-size: 0.95rem; padding: 0.25rem 0.6rem; }}
.tr-hero .tr-line {{ color: #E2E8F0; font-size: 0.9rem; margin-top: 0.7rem; line-height: 1.5; }}
.tr-hero .tr-foot {{ color: #94A3B8; font-size: 0.8rem; margin-top: 0.7rem; line-height: 1.45;
  border-top: 1px solid #1E293B; padding-top: 0.6rem; }}
.st-key-panel_gaps {{ animation: tr-rise 0.6s ease-out; }}
@keyframes tr-rise {{ from {{ transform: translateY(10px); }} to {{ transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{
  .tr-track, .tr-hero, .st-key-panel_gaps {{ animation: none; }}
}}
.tr-intro {{ color: #4B5563; font-size: 0.95rem; margin: 0 0 0.9rem 0; }}
/* Accueil : ce que fait le site, en une phrase, et la légende du signal. */
.tr-welcome {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1.6rem;
  margin: 0 0 0.9rem 0; }}
.tr-welcome-text {{ flex: 1 1 32rem; color: #1F2937; font-size: 0.98rem; line-height: 1.5; }}
.tr-legend {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; color: #4B5563;
  font-size: 0.82rem; }}
.tr-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
/* Carte « Is the signal reliable? » sous la prochaine enchère. */
.tr-trust {{ padding: 0 0.9rem 0.2rem 0.9rem; }}
.tr-trust-row {{ display: flex; gap: 0.8rem; align-items: baseline; padding: 0.35rem 0;
  font-size: 0.86rem; color: #374151; line-height: 1.45; }}
.tr-trust-row + .tr-trust-row {{ border-top: 1px solid #F1F3F6; }}
.tr-trust-value {{ font-size: 1.5rem; font-weight: 700; color: #111827; min-width: 3.6rem;
  white-space: nowrap;
  font-variant-numeric: tabular-nums; }}
.tr-panel {{ background: #FFFFFF; border: 1px solid #E2E6EC; border-radius: 6px;
  margin-bottom: 0.9rem; overflow-x: auto; }}
.tr-panel-head {{ font-weight: 600; font-size: 1.05rem; padding: 0.7rem 0.9rem 0.5rem 0.9rem;
  color: #111827; }}
.tr-panel-note {{ color: #6B7280; font-size: 0.8rem; padding: 0.5rem 0.9rem 0.7rem 0.9rem; }}
details.tr-help > summary {{ list-style: none; cursor: pointer; display: flex;
  justify-content: space-between; align-items: baseline; gap: 0.6rem; }}
details.tr-help > summary::-webkit-details-marker {{ display: none; }}
.tr-how {{ color: {PRIMARY}; font-size: 0.78rem; font-weight: 500; white-space: nowrap; }}
.tr-how::before {{ content: "ⓘ "; }}
details.tr-help[open] .tr-how {{ color: #6B7280; }}
.tr-help-body {{ margin: 0 0.9rem 0.7rem 0.9rem; padding: 0.55rem 0.8rem 0.55rem 1.7rem;
  background: #EFF6FF; border-radius: 4px; color: #1F2937; font-size: 0.82rem;
  line-height: 1.45; }}
.tr-help-body li {{ margin: 0.15rem 0; font-size: 0.82rem !important; line-height: 1.45; }}
[class*="st-key-panel"] .tr-panel-head {{ padding: 0 0 0.3rem 0; }}
[class*="st-key-panel"] .tr-help-body {{ margin: 0.2rem 0 0.4rem 0; }}
.tr-reading {{ font-size: 0.9rem; color: #1F2937; line-height: 1.5; margin-bottom: 0.5rem; }}
.tr-reading .tr-tag {{ margin-right: 0.35rem; }}
.tr-small {{ color: #6B7280; font-size: 0.8rem; line-height: 1.45; }}
.tr-body {{ padding: 0 0.9rem 0.7rem 0.9rem; color: #1F2937; }}
.tr-body p, .tr-body li {{ font-size: 0.9rem !important; line-height: 1.55; }}
.tr-body p {{ margin: 0 0 0.3rem 0; }}
.tr-body ul {{ margin: 0; padding-left: 1.2rem; }}
table.tr-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem;
  font-variant-numeric: tabular-nums; }}
table.tr-table th {{ background: #F3F5F8; color: #374151; font-weight: 600; font-size: 0.8rem;
  padding: 0.45rem 0.55rem; border-top: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB;
  white-space: nowrap; }}
table.tr-table td {{ padding: 0.42rem 0.55rem; border-bottom: 1px solid #F1F3F6;
  white-space: nowrap; }}
table.tr-table td.tr-wrap {{ white-space: normal; }}
.tr-scroll {{ max-height: 30rem; overflow-y: auto; }}
.tr-scroll table.tr-table th {{ position: sticky; top: 0; z-index: 1; }}
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
.tr-better-text {{ color: {UP}; }}
.tr-worse-text {{ color: {ALERT}; }}
.tr-unclear-text {{ color: #374151; }}
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
  /* Téléphone : un en-tête court, onglets sur une ligne que l'on fait défiler au doigt. */
  .tr-topline .tr-tagline {{ display: none; }}
  .tr-topline .tr-meta {{ font-size: 0.78rem; }}
  .st-key-topbar {{ padding: 0.5rem 0.7rem 0.1rem 0.7rem; }}
  .st-key-topbar [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: 0.5rem; }}
  .st-key-topbar [data-testid="stColumn"] {{ min-width: 0 !important; }}
  .st-key-topbar [data-testid="stColumn"]:last-child {{ flex: 0 0 auto !important;
    width: auto !important; }}
  .st-key-topbar [data-testid="stColumn"]:last-child button p {{ display: none; }}
  .st-key-topnav {{ flex-wrap: nowrap !important; overflow-x: auto; scrollbar-width: none; }}
  .st-key-topnav::-webkit-scrollbar {{ display: none; }}
  .tr-welcome-text {{ font-size: 0.92rem; }}
}}
</style>
"""


def page_style() -> None:
    """Style du tableau de bord pour toutes les pages."""
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


# Logo de CoverCast (le même que sur les slides) : deux barres pleines pour les enchères passées
# et une barre en pointillés pour la prévision.
LOGO = (
    '<svg class="tr-logo" aria-label="CoverCast logo" width="30" height="30" viewBox="0 0 76 76">'
    '<rect x="0" y="0" width="76" height="76" rx="18" fill="#2563EB"/>'
    '<rect x="15" y="40" width="12" height="22" rx="2" fill="#FFFFFF"/>'
    '<rect x="32" y="28" width="12" height="34" rx="2" fill="#FFFFFF"/>'
    '<rect x="50.5" y="17.5" width="11" height="43" rx="2" fill="none" stroke="#FFFFFF"'
    ' stroke-width="3" stroke-dasharray="5 4"/></svg>'
)


def top_bar(pages: list | None = None, current: object | None = None) -> None:
    """En-tête du site, sur chaque page : nom et logo, heure de la dernière mise à jour, alerte
    éventuelle sur les sources, bouton de mise à jour, puis les onglets des pages."""
    status = json_file(live.STATUS_FILE)
    quality = json_file("data_quality.json")
    fetch = json_file(FETCH_STATUS_FILE)
    problems = []
    if fetch and ad.failed_sources(fetch):
        problems.append(f"{len(ad.failed_sources(fetch))} source(s) failed to update")
    if quality and quality.get("status") == "rouge":
        problems.append("data check failed")
    alert = (
        f'<a class="tr-alert" href="qualite" target="_self">⚠ {escape("; ".join(problems))}</a>'
        if problems
        else ""
    )
    with st.container(key="topbar"):
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.markdown(
            f'<div class="tr-topline">{LOGO}'
            '<span class="tr-brand">Cover<span class="tr-cast">Cast</span></span>'
            '<span class="tr-tagline">Demand forecasts for US Treasury auctions</span>'
            f'<span class="tr-meta">Data updated {escape(moment_en(status.get("run_at")))}</span>'
            f"{alert}</div>",
            unsafe_allow_html=True,
        )
        clicked = right.button(
            "Refresh now",
            icon=":material/refresh:",
            width="stretch",
            help="Reloads the auction calendar and today's yields, then recomputes provisional"
            " forecasts with the existing models. Official forecasts are recorded each evening.",
        )
        if pages:
            with st.container(horizontal=True, key="topnav", gap="small"):
                for item in pages:
                    active = current is not None and item.url_path == current.url_path
                    # Onglet actif : souligné, dans un conteneur nommé que le CSS reconnaît.
                    with st.container(key=f"nav_{'on' if active else 'off'}_{item.url_path}"):
                        st.page_link(item, label=item.title)
    if clicked:  # les messages s'affichent sous le bandeau, pas dedans
        _refresh()


# --- Petits morceaux de HTML pour les panneaux (textes déjà échappés par l'appelant) --------


def panel_head(title: str, guide: list[str] | None = None) -> str:
    """Titre d'un panneau. Avec `guide`, le titre s'ouvre au clic sur une courte aide
    « How to read » (élément HTML <details> : marche sans JavaScript, au doigt comme à la
    souris)."""
    if not guide:
        return f'<div class="tr-panel-head">{escape(title)}</div>'
    items = "".join(f"<li>{item}</li>" for item in guide)
    return (
        f'<details class="tr-help"><summary class="tr-panel-head">{escape(title)}'
        f'<span class="tr-how">How to read</span></summary>'
        f'<ul class="tr-help-body">{items}</ul></details>'
    )


def panel(title: str, body: str, note: str = "", guide: list[str] | None = None) -> str:
    """Un panneau blanc : titre (et son aide), contenu, note grise facultative."""
    foot = f'<div class="tr-panel-note">{note}</div>' if note else ""
    return f'<div class="tr-panel">{panel_head(title, guide)}{body}{foot}</div>'


def table(
    headers: list[str],
    rows: list[list[str | tuple[str, str]]],
    right: set[int] | None = None,
    wrap: set[int] | None = None,
    scroll: bool = False,
) -> str:
    """Tableau compact ; `right` : colonnes alignées à droite (nombres), `wrap` : colonnes dont
    le texte peut passer à la ligne (les autres restent sur une ligne), `scroll` : tableau long
    qui défile sous son en-tête. Une cellule peut être un couple (texte, classe CSS)."""
    right, wrap = right or set(), wrap or set()

    def cell(tag: str, index: int, content: str | tuple[str, str]) -> str:
        extra = ""
        if isinstance(content, tuple):  # (texte, classe CSS en plus)
            content, extra = content
        flags = (("tr-right", index in right), ("tr-wrap", index in wrap), (extra, bool(extra)))
        classes = " ".join(name for name, on in flags if on)
        return (
            f'<{tag} class="{classes}">{content}</{tag}>'
            if classes
            else f"<{tag}>{content}</{tag}>"
        )

    head = "".join(cell("th", i, escape(h)) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows
    )
    html = f'<table class="tr-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return f'<div class="tr-scroll">{html}</div>' if scroll else html


def signed(value: float, text: str) -> str:
    """Texte coloré selon le signe de la valeur : vert si elle monte, rouge si elle baisse."""
    css = "tr-up" if value > 0 else "tr-down" if value < 0 else "tr-flat"
    return f'<span class="{css}">{escape(text)}</span>'


def signed_number(value: float, decimals: int, suffix: str = "") -> str:
    """Nombre signé et coloré, arrondi d'abord : +0,003 s'affiche « 0.00 » en gris, pas
    « +0.00 » en vert."""
    if pd.isna(value):
        return en(value)
    shown = round(float(value), decimals) + 0.0  # + 0.0 : pas de « −0 »
    return signed(shown, en(shown, decimals, sign=shown != 0) + suffix)


def demand_tag(signal: str) -> str:
    """Étiquette colorée du signal de demande."""
    label, css = {ad.WEAK: ("Weaker", "tr-weak"), ad.STRONG: ("Stronger", "tr-strong"),
                  ad.NORMAL: ("Normal", "tr-normal")}.get(signal, ("n/a", "tr-normal"))  # fmt: skip
    return f'<span class="tr-tag {css}">{label}</span>'


def page_header(title: str, intro: str) -> None:
    """Titre de la page et une phrase d'introduction (HTML déjà échappé par l'appelant)."""
    st.title(title)
    st.markdown(f'<p class="tr-intro">{intro}</p>', unsafe_allow_html=True)


def box(name: str, title: str, guide: list[str] | None = None) -> object:
    """Panneau blanc qui peut contenir des éléments Streamlit (graphiques, menus) : son titre et
    son aide « How to read » sont écrits, le conteneur est rendu pour la suite (`with`)."""
    container = st.container(border=True, key=f"panel_{name}")
    container.markdown(panel_head(title, guide), unsafe_allow_html=True)
    return container


def stats(items: list[tuple[str, str]], three: bool = False) -> str:
    """Tuiles de chiffres : (valeur, texte) ; textes déjà échappés par l'appelant."""
    tiles = "".join(
        f'<div class="tr-stat"><div class="tr-value">{value}</div>'
        f'<div class="tr-text">{text}</div></div>'
        for value, text in items
    )
    return f'<div class="tr-stats{" tr-three" if three else ""}">{tiles}</div>'


def small(text: str) -> None:
    """Petite note grise sous un graphique (HTML déjà échappé par l'appelant)."""
    st.markdown(f'<div class="tr-small">{text}</div>', unsafe_allow_html=True)


def upcoming_guide(more: bool = False) -> list[str]:
    """Aide « How to read » du tableau des enchères à venir (accueil et page détaillée)."""
    threshold = en(SIGNAL_THRESHOLD, 2)
    guide = [
        "<b>Forecast</b>: bid-to-cover expected by the average of 5 models, using only data"
        " public the evening before.",
        "<b>Usual</b>: average bid-to-cover of the last 3 auctions of the same maturity.",
        f"<b>Difference</b> = Forecast − Usual. <b>Weaker</b> if −{threshold} or less,"
        f" <b>Stronger</b> if +{threshold} or more, otherwise <b>Normal</b>.",
        "<b>Size (est.)</b>: not announced yet, so the size of the previous auction is used.",
    ]
    if more:
        guide.append(
            "<b>Dealer share, Indirect share, Tail</b>: forecasts of the other measures. The"
            " Model scorecard on Home shows how reliable each one is; the signal uses the"
            " bid-to-cover only."
        )
    return guide


def upcoming_html(frame: pd.DataFrame, more: bool = False) -> str:
    """Tableau des enchères à venir ; `more` ajoute les prévisions des trois autres mesures."""
    headers = ["Date", "Auction", "Size", "Forecast", "Usual", "Difference", "Demand"]
    if more:
        headers += ["Dealer share", "Indirect share", "Tail"]
    rows = []
    for row in frame.itertuples():
        cells = [
            escape(date_en(row.auction_date, year=False)),
            f"<b>{escape(row.title)}</b>",
            escape(ad.amount(row.offering_amt, bool(row.offering_estimated))),
            en(row.pred_btc, 2),
            en(row.bench_mean3_btc, 2),
            signed_number(row.gap_btc, 3),  # 3 décimales : le signal porte sur l'écart exact
            demand_tag(row.signal),
        ]
        if more:
            cells += [
                ad.format_target(row.pred_dealer_share, "dealer_share"),
                ad.format_target(row.pred_indirect_share, "indirect_share"),
                ad.format_target(row.pred_tail_bp, "tail_bp"),
            ]
        rows.append(cells)
    right = {2, 3, 4, 5} | ({7, 8, 9} if more else set())
    return table(headers, rows, right=right, wrap={2})  # « (est.) » passe à la ligne si besoin


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


def _refresh() -> None:
    from tresor import refresh

    with st.spinner("Downloading and computing…"):
        try:
            predictions, failure = refresh.light_refresh()
        except refresh.RefreshBusy:
            st.info("A refresh is already running. Try again in a minute.")
            return
        except refresh.ModelsUnavailable:
            st.warning("Models are not available right now. Please try again later.")
            return
        except Exception as error:  # l'application reste utilisable
            st.error(f"Refresh failed: {error}")
            return
    st.session_state["provisional"] = predictions
    message = "Provisional forecast ready: see Upcoming auctions."
    if failure:
        st.warning(f"Partial refresh ({failure}). {message}")
    else:
        st.success(message)
