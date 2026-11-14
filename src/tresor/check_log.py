"""Contrôle du passage du soir : aucune ligne figée du journal n'a changé, disparu ou été
ajoutée après coup.

`python -m tresor.check_log ancien.parquet nouveau.parquet` sort avec le code 1 si une
prévision dont l'enchère est aujourd'hui ou passée (date de New York) a été modifiée,
supprimée ou ajoutée, ou si une prévision est datée du jour de son enchère ou après. Le
workflow lance ce contrôle avant d'enregistrer les données : la garantie du journal figé tient
alors quel que soit le programme qui a écrit le fichier.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tresor import live

# Le statut du passage du soir n'est cru que s'il vient d'être écrit (voir `frozen_today`).
STATUS_MAX_AGE = pd.Timedelta(hours=6)


def frozen_today(new_path: Path, now: pd.Timestamp) -> pd.Timestamp:
    """Date de New York qui décide quelles lignes sont figées.

    C'est la date du passage du soir qui vient d'écrire le journal (`live_status.json`, à côté
    de lui) : le contrôle juge alors avec la même date que l'écriture, même si minuit est passé
    à New York entre les deux. Un statut absent, illisible ou ancien est ignoré : on prend la
    date de l'instant présent.
    """
    try:
        status = json.loads((new_path.parent / live.STATUS_FILE).read_text(encoding="utf-8"))
        run_at = pd.Timestamp(status["run_at"])
        if run_at.tzinfo is not None and pd.Timedelta(0) <= now - run_at <= STATUS_MAX_AGE:
            return live.new_york_today(run_at)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return live.new_york_today(now)


def check(old_path: Path, new_path: Path, now: pd.Timestamp | None = None) -> str | None:
    """Message d'erreur si le journal figé n'est pas respecté, sinon rien."""
    if not old_path.exists():
        return None  # premier passage : rien à comparer
    if not new_path.exists():
        return f"{new_path.name} a disparu"
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    try:
        live.check_frozen_unchanged(
            live.read_log(old_path), live.read_log(new_path), frozen_today(new_path, now)
        )
    except (RuntimeError, ValueError) as error:
        return str(error)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage : python -m tresor.check_log ancien.parquet nouveau.parquet")
        return 2
    problem = check(Path(arguments[0]), Path(arguments[1]))
    if problem:
        print(f"Journal figé : {problem}")
        return 1
    print("Journal figé : aucune ligne figée n'a changé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
