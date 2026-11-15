"""Supports du rapport et de l'oral : figures et chiffres clés, depuis les fichiers de data/.

    python reports/make_figures.py           # figures (reports/figures/) et chiffres_cles.md
    python reports/make_figures.py --check   # vérifie chiffres_cles.md contre les fichiers

Rien n'est recalculé par les modèles : tout est relu dans data/ (backtest et stratégie produits
par `python -m tresor.pipeline --backtest --strategy`, données mises à jour chaque soir).
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
from tresor.config import DATA_DIR, FIGURES_DIR  # noqa: E402
from tresor.strategy import COST_BP  # noqa: E402

KEY_FIGURES_FILE = ROOT / "reports" / "chiffres_cles.md"


def make_figures(data_dir: Path = DATA_DIR, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Toutes les figures, relues depuis les fichiers de data/."""
    output = data_dir / "output"
    quality = json.loads((output / "data_quality.json").read_text(encoding="utf-8"))
    quintiles = pd.read_csv(output / "quintiles.csv")
    market = pd.read_parquet(data_dir / "clean" / "market.parquet")
    return [
        plots.source_coverage(quality, SOURCE_LABELS, figures_dir),
        plots.auctions_by_year(
            pd.read_csv(output / "auctions_by_family_year.csv"),
            key_figures.auctions_overview(data_dir),
            figures_dir,
        ),
        plots.publication_delays(
            key_figures.publication_delays(market), key_figures.DELAYS_SINCE, figures_dir
        ),
        *plots.all_figures(
            pd.read_csv(output / "backtest_metrics.csv"),
            pd.read_parquet(output / "backtest_predictions.parquet"),
            pd.read_csv(output / "new_sources_comparison.csv"),
            pd.read_csv(output / "best_models.csv"),
            pd.read_csv(output / "feature_importance.csv"),
            figures_dir,
        ),
        plots.quintiles(quintiles[quintiles["period"] == "2009-2026"], figures_dir),
        plots.strategy_pnl(
            pd.read_parquet(output / "strategy_trades.parquet"), COST_BP, figures_dir
        ),  # fmt: skip
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="vérifier chiffres_cles.md")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--key-figures", type=Path, default=KEY_FIGURES_FILE)
    args = parser.parse_args(argv)
    if args.check:
        errors, changes = key_figures.check(args.key_figures, args.data_dir)
        for change in changes:
            print(f"Changé depuis l'écriture (normal, données plus récentes) : {change}")
        for error in errors:
            print(f"ERREUR : {error}")
        print(f"{len(key_figures.KEY_FIGURES)} chiffres recalculés : {len(errors)} erreur(s).")
        return 1 if errors else 0
    for path in make_figures(args.data_dir, args.figures_dir):
        print(f"Figure : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    args.key_figures.write_text(key_figures.render(args.data_dir), encoding="utf-8")
    print(f"Chiffres clés : {len(key_figures.KEY_FIGURES)} chiffres dans {args.key_figures.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
