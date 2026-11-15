"""Supports de la phase 11 : chaque chiffre clé est recalculé depuis les fichiers de data/."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from tests.test_live import EMPTY, predictions, utc

from tresor import key_figures as kf
from tresor import live
from tresor.config import DATA_DIR, OUTPUT_DIR
from tresor.strategy import COST_BP

ROOT = Path(__file__).resolve().parents[1]
KEY_FIGURES_FILE = ROOT / "reports" / "chiffres_cles.md"
pytestmark = pytest.mark.skipif(
    not (OUTPUT_DIR / "backtest_metrics.csv").exists(), reason="données du dépôt absentes"
)

spec = importlib.util.spec_from_file_location("make_figures", ROOT / "reports" / "make_figures.py")
assert spec is not None and spec.loader is not None
make_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_figures)

REFERENCE_R2 = "R² contre mean3, notes et bonds, bid-to-cover, 2020-2026"
CLEANED = "Enchères gardées après nettoyage, annoncées comprises"

# Fichiers réécrits chaque soir par le passage automatique (--fetch --clean --predict) ; les
# autres fichiers de sortie ne sont écrits que par --backtest --strategy, lancés à la main.
NIGHTLY = {
    "output/cleaning_report.json",
    "output/data_quality.json",
    "output/features_missing.csv",
    "output/auctions_by_family_year.csv",
    "output/live_status.json",
    "output/predictions_log.parquet",
    "output/live_tracking.parquet",
    "output/fetch_status.json",
}


def row_of(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {label} |"))


def written(tmp_path: Path, text: str | None = None, **values: str) -> Path:
    """chiffres_cles.md écrit depuis les fichiers du dépôt, avec des valeurs remplacées."""
    text = kf.render() if text is None else text
    for label, value in values.items():
        line = row_of(text, label)
        cells = line.split(" | ")
        cells[1] = value
        text = text.replace(line, " | ".join(cells))
    path = tmp_path / "chiffres_cles.md"
    path.write_text(text, encoding="utf-8")
    return path


def other_data(text: str) -> str:
    """Le même fichier, comme s'il avait été écrit avec les données d'un autre soir."""
    stamp = next(line for line in text.splitlines() if line.startswith(kf.STAMP))
    return text.replace(stamp, f"{kf.STAMP}contrôle qualité d'un autre soir.")


def test_the_committed_key_figures_match_the_files() -> None:
    """Critère de la phase 11 : chaque chiffre de chiffres_cles.md est recalculé. Avec les
    données du soir où le fichier a été écrit, les chiffres du jour sont vérifiés aussi."""
    errors, changes = kf.check(KEY_FIGURES_FILE)

    assert errors == []
    stamp = kf.STAMP + kf.data_stamp()
    if stamp in KEY_FIGURES_FILE.read_text(encoding="utf-8"):
        assert changes == []
    assert len(kf.read_values(KEY_FIGURES_FILE.read_text(encoding="utf-8"))) == len(kf.KEY_FIGURES)


def test_values_are_those_of_the_files() -> None:
    values = kf.compute_all()
    metrics = pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv")
    row = metrics[
        (metrics["model"] == "moyenne_5")
        & (metrics["feature_set"] == "toutes")
        & (metrics["period"] == "2020-2026")
        & (metrics["family"] == "nominal_coupon")
        & (metrics["target"] == "btc")
    ].iloc[0]

    expected = (
        f"{100 * row['r2_vs_mean3']:+.1f} % [{100 * row['r2_ci_low']:.1f} ;"
        f" {100 * row['r2_ci_high']:.1f}]"
    )
    assert values[REFERENCE_R2] == expected.replace(".", ",")
    report = json.loads((OUTPUT_DIR / "cleaning_report.json").read_text())["auctions"]
    assert values[CLEANED].replace("\xa0", "") == str(report["rows_out"])


def test_each_figure_has_the_nature_of_its_file() -> None:
    for figure in kf.KEY_FIGURES:
        nightly = figure.source.startswith("clean/") or figure.source in NIGHTLY
        assert figure.nature == (kf.DAILY if nightly else kf.FIXED), figure.label


def test_a_changed_fixed_figure_is_always_an_error(tmp_path: Path) -> None:
    for text in (kf.render(), other_data(kf.render())):
        errors, changes = kf.check(written(tmp_path, text, **{REFERENCE_R2: "+99,0 %"}))

        assert any(REFERENCE_R2 in error for error in errors) and changes == []


def test_a_changed_daily_figure_is_an_error_with_the_same_data(tmp_path: Path) -> None:
    errors, changes = kf.check(written(tmp_path, **{CLEANED: "1"}))

    assert len(errors) == 1 and CLEANED in errors[0] and changes == []


def test_a_changed_daily_figure_is_only_reported_with_newer_data(tmp_path: Path) -> None:
    errors, changes = kf.check(written(tmp_path, other_data(kf.render()), **{CLEANED: "1"}))

    assert errors == [] and len(changes) == 1 and CLEANED in changes[0]


def test_any_other_change_to_the_file_is_an_error(tmp_path: Path) -> None:
    text = kf.render()
    line = row_of(text, REFERENCE_R2)
    cases = {
        "nature": text.replace(line, line.replace("| figé |", "| du jour |")),
        "calcul": text.replace(line, line.replace("r2_vs_mean3 [", "r2_vs_constant [")),
        "double": text.replace(line, line.replace(line.split(" | ")[1], "+99,0 %") + "\n" + line),
        "glossaire": text.replace("**Petit glossaire.**", "**Glossaire.**"),
        "absent": text.replace(line + "\n", ""),
        "inventé": text.replace(line, line.replace(REFERENCE_R2, "Chiffre inventé")),
    }
    for name, changed in cases.items():
        errors, _ = kf.check(written(tmp_path, changed))

        assert errors, name


def test_a_missing_file_is_reported_not_a_crash(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR / "output", data / "output")
    shutil.copytree(DATA_DIR / "clean", data / "clean")
    (data / "output" / "quintiles.csv").unlink()

    errors, _ = kf.check(KEY_FIGURES_FILE, data)

    assert len(errors) == 1 and "recalcul impossible" in errors[0]


def test_a_figure_needs_exactly_one_row() -> None:
    table = pd.DataFrame({"period": ["a", "a", "b"], "value": [1, 2, 3]})

    assert kf._row(table, period="b")["value"] == 3
    for period in ("a", "c"):  # deux lignes, ou aucune : pas de chiffre au hasard
        with pytest.raises(ValueError):
            kf._row(table, period=period)


def test_the_auctions_figure_numbers() -> None:
    overview = kf.auctions_overview()
    report = json.loads((OUTPUT_DIR / "cleaning_report.json").read_text())["auctions"]
    features = pd.read_parquet(DATA_DIR / "clean" / "features.parquet")
    completed = features[
        (features["status"] == "completed") & (features["auction_date"] >= "2009-01-01")
    ]
    upcoming = features[features["status"] == "upcoming"]

    assert overview["kept"] == report["rows_out"]
    assert overview["sample"] == len(completed)  # features ne contient que bills, notes, bonds
    assert overview["upcoming"] == len(upcoming)
    assert overview["last_completed"] == completed["auction_date"].max()
    assert overview["last_upcoming"] == upcoming["auction_date"].max()


def test_publication_delays_are_measured_from_the_dates() -> None:
    market = pd.DataFrame(
        {
            "source": ["cboe_vix", "cboe_vix", "fred_series", "fred_series", "cboe_vix"],
            "series": ["vix_close", "vix_close", "fred_dtwexbgs", "fred_walcl", "vix_close"],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-02", "2010-01-04"]
            ),  # fmt: skip
            "available_at": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-10", "2020-01-03", "2010-03-01"]
            ),  # fmt: skip
        }
    )

    table = kf.publication_delays(market).set_index("label")

    assert table.loc["CBOE : VIX", "median"] == 0  # la valeur de 2010 est avant 2014
    assert table.loc["FRED : indice dollar", "median"] == 8
    assert table.loc["FRED : autres séries", "median"] == 1
    assert table.loc["FRED : indice dollar", "note"] != ""
    assert list(table.index) == ["CBOE : VIX", "FRED : autres séries", "FRED : indice dollar"]


def test_the_script_writes_every_figure_and_checks_them(tmp_path: Path) -> None:
    key_figures = tmp_path / "chiffres_cles.md"
    arguments = ["--figures-dir", str(tmp_path / "figures"), "--key-figures", str(key_figures)]

    assert make_figures.main(arguments) == 0
    names = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert names == sorted(path.name for path in (ROOT / "reports" / "figures").glob("*.png"))
    assert make_figures.main([*arguments, "--check"]) == 0
    written(tmp_path, **{REFERENCE_R2: "+99,0 %"})
    assert make_figures.main([*arguments, "--check"]) == 1


def test_the_readme_numbers_are_key_figures() -> None:
    """Chaque valeur avec intervalle citée dans les résultats du README vient de
    chiffres_cles.md (même texte, mêmes arrondis)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = readme.split("## Résultats")[1].split("\n## ")[0]
    quoted = re.findall(r"[+−]?\d[\d,]* (?:%|pb) \[[^\]]+\]", results)
    values = kf.compute_all().values()

    assert len(quoted) >= 7
    for text in quoted:
        assert any(value.replace("\xa0", " ") == text for value in values), text


def test_live_counts_on_a_small_log(tmp_path: Path) -> None:
    """Journal de 2 enchères : celle du jour est figée, une seule a son résultat publié."""
    output = tmp_path / "output"
    output.mkdir()
    today = pd.Timestamp("2026-11-18")
    first = predictions(2.5, auction=today, cusip="A")
    first = pd.concat([first, first.assign(model="lasso")])  # deux lignes pour une enchère
    second = predictions(2.6, auction=today + pd.Timedelta(days=2), cusip="B")
    log = live.update_log(EMPTY, pd.concat([first, second]), utc("2026-11-17 23:17"))
    log.to_parquet(output / live.LOG_FILE, index=False)
    (output / live.STATUS_FILE).write_text(json.dumps({"today_new_york": "2026-11-18"}))
    status = pd.Series(["realisee", "en_attente"], index=["A", "B"])
    log.assign(status=log["cusip"].map(status)).to_parquet(output / live.TRACKING_FILE)

    assert kf._log_counts(tmp_path, "lignes") == str(len(log))
    assert kf._log_counts(tmp_path, "figées") == "1"  # le jour de l'enchère compte
    assert kf._log_counts(tmp_path, "réalisées") == "1"


def test_the_script_passes_the_right_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le script lit le dossier de données demandé et le coût de la stratégie du code."""
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR / "output", data / "output")
    shutil.copytree(DATA_DIR / "clean", data / "clean")
    report_path = data / "output" / "cleaning_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["auctions"]["rows_out"] = 123456
    report_path.write_text(json.dumps(report), encoding="utf-8")
    seen: dict[str, object] = {}

    def strategy_pnl(trades: pd.DataFrame, cost: float, folder: Path) -> Path:
        seen["cost"] = cost
        return folder / "strategie_pnl_cumule.png"

    def auctions_by_year(table: pd.DataFrame, overview: dict, folder: Path) -> Path:
        seen["overview"] = overview
        return folder / "donnees_encheres_par_an.png"

    monkeypatch.setattr(make_figures.plots, "strategy_pnl", strategy_pnl)
    monkeypatch.setattr(make_figures.plots, "auctions_by_year", auctions_by_year)
    key_figures = tmp_path / "chiffres_cles.md"

    make_figures.main(["--data-dir", str(data), "--figures-dir", str(tmp_path / "figures"),
                       "--key-figures", str(key_figures)])  # fmt: skip

    assert seen["cost"] == COST_BP
    assert seen["overview"]["kept"] == 123456
    assert "| 123\xa0456 |" in key_figures.read_text(encoding="utf-8")
