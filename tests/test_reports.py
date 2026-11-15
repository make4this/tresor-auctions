"""Supports de la phase 11 : chaque chiffre clé est recalculé depuis les fichiers de sortie."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tresor import key_figures as kf
from tresor.config import OUTPUT_DIR

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
CLEANED = "Enchères gardées après nettoyage"


def written(tmp_path: Path, replace: dict[str, str] | None = None) -> Path:
    """chiffres_cles.md écrit depuis les fichiers du dépôt, avec des valeurs remplacées."""
    text = kf.render()
    for label, value in (replace or {}).items():
        line = next(line for line in text.splitlines() if line.startswith(f"| {label} |"))
        old = line.split(" | ")[1]
        text = text.replace(line, line.replace(f"| {old} |", f"| {value} |", 1))
    path = tmp_path / "chiffres_cles.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_committed_key_figures_match_the_output_files() -> None:
    """Critère de la phase 11 : chaque chiffre de chiffres_cles.md est recalculé."""
    errors, _ = kf.check(KEY_FIGURES_FILE)

    assert errors == []
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

    expected = f"{row['r2_vs_mean3']:.2f} [{row['r2_ci_low']:.2f} ; {row['r2_ci_high']:.2f}]"
    assert values[REFERENCE_R2] == expected.replace(".", ",")
    assert values["Sources publiques collectées"] == "11"


def test_a_changed_fixed_figure_is_an_error(tmp_path: Path) -> None:
    path = written(tmp_path, {REFERENCE_R2: "0,99 [0,98 ; 1,00]"})

    errors, changes = kf.check(path)

    assert len(errors) == 1 and REFERENCE_R2 in errors[0] and changes == []


def test_a_changed_daily_figure_is_only_reported(tmp_path: Path) -> None:
    path = written(tmp_path, {CLEANED: "1"})

    errors, changes = kf.check(path)

    assert errors == [] and len(changes) == 1 and CLEANED in changes[0]


def test_missing_or_unknown_figures_are_errors(tmp_path: Path) -> None:
    text = kf.render()
    line = next(line for line in text.splitlines() if line.startswith(f"| {REFERENCE_R2} |"))
    path = tmp_path / "chiffres_cles.md"
    path.write_text(text.replace(line, line.replace(REFERENCE_R2, "Chiffre inventé")))

    errors, _ = kf.check(path)

    assert any("absent" in error and REFERENCE_R2 in error for error in errors)
    assert any("inconnu" in error and "Chiffre inventé" in error for error in errors)


def test_a_figure_needs_exactly_one_row() -> None:
    table = pd.DataFrame({"period": ["a", "a", "b"], "value": [1, 2, 3]})

    assert kf._row(table, period="b")["value"] == 3
    for period in ("a", "c"):  # deux lignes, ou aucune : pas de chiffre au hasard
        with pytest.raises(ValueError):
            kf._row(table, period=period)


def test_the_script_writes_every_figure_and_checks_them(tmp_path: Path) -> None:
    key_figures = tmp_path / "chiffres_cles.md"
    arguments = ["--figures-dir", str(tmp_path / "figures"), "--key-figures", str(key_figures)]

    assert make_figures.main(arguments) == 0
    names = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert names == sorted(path.name for path in (ROOT / "reports" / "figures").glob("*.png"))
    assert make_figures.main([*arguments, "--check"]) == 0
    written(tmp_path, {REFERENCE_R2: "0,99 [0,98 ; 1,00]"})
    assert make_figures.main([*arguments, "--check"]) == 1
