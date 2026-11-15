"""Supports du rapport et de l'oral : figures et chiffres clés, depuis les fichiers de sortie.

    python reports/make_figures.py           # figures (reports/figures/) et chiffres_cles.md
    python reports/make_figures.py --check   # vérifie chiffres_cles.md contre les fichiers

Rien n'est recalculé par les modèles : tout est relu dans data/output/ (backtest et stratégie
produits par `python -m tresor.pipeline --backtest --strategy`, données mises à jour chaque soir).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # utilisable sans installer le paquet
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tresor import key_figures, plots  # noqa: E402
from tresor.app_data import SOURCE_LABELS  # noqa: E402
from tresor.config import FIGURES_DIR, OUTPUT_DIR  # noqa: E402

KEY_FIGURES_FILE = ROOT / "reports" / "chiffres_cles.md"


def make_figures(output_dir: Path = OUTPUT_DIR, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Toutes les figures, relues depuis les fichiers de sortie."""
    quality = json.loads((output_dir / "data_quality.json").read_text(encoding="utf-8"))
    quintiles = pd.read_csv(output_dir / "quintiles.csv")
    paths = [
        plots.source_coverage(quality, SOURCE_LABELS, figures_dir),
        plots.auctions_by_year(
            pd.read_csv(output_dir / "auctions_by_family_year.csv"),
            str(quality["today_new_york"]),
            figures_dir,
        ),
        *plots.all_figures(
            pd.read_csv(output_dir / "backtest_metrics.csv"),
            pd.read_parquet(output_dir / "backtest_predictions.parquet"),
            pd.read_csv(output_dir / "new_sources_comparison.csv"),
            pd.read_csv(output_dir / "best_models.csv"),
            pd.read_csv(output_dir / "feature_importance.csv"),
            figures_dir,
        ),
        plots.quintiles(quintiles[quintiles["period"] == "2009-2026"], figures_dir),
        plots.strategy_pnl(
            pd.read_parquet(output_dir / "strategy_trades.parquet"), 0.5, figures_dir
        ),  # fmt: skip
    ]
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="vérifier chiffres_cles.md")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--key-figures", type=Path, default=KEY_FIGURES_FILE)
    args = parser.parse_args(argv)
    if args.check:
        errors, changes = key_figures.check(args.key_figures, args.output_dir)
        for change in changes:
            print(f"Changé depuis l'écriture (normal, chiffre du jour) : {change}")
        for error in errors:
            print(f"ERREUR : {error}")
        print(f"{len(key_figures.KEY_FIGURES)} chiffres recalculés : {len(errors)} erreur(s).")
        return 1 if errors else 0
    for path in make_figures(args.output_dir, args.figures_dir):
        print(f"Figure : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    args.key_figures.write_text(key_figures.render(args.output_dir), encoding="utf-8")
    print(f"Chiffres clés : {len(key_figures.KEY_FIGURES)} chiffres dans {args.key_figures.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
