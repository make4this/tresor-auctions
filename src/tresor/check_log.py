"""Contrôle du passage du soir : aucune ligne figée du journal n'a changé ni disparu.

`python -m tresor.check_log ancien.parquet nouveau.parquet` sort avec le code 1 si une
prévision dont l'enchère est aujourd'hui ou passée (date de New York) a été modifiée ou
supprimée. Le workflow lance ce contrôle avant d'enregistrer les données : la garantie du
journal figé tient alors quel que soit le programme qui a écrit le fichier.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tresor import live


def check(old_path: Path, new_path: Path, now: pd.Timestamp | None = None) -> str | None:
    """Message d'erreur si une ligne figée a changé, sinon rien."""
    if not old_path.exists():
        return None  # premier passage : rien à comparer
    if not new_path.exists():
        return f"{new_path.name} a disparu"
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    try:
        live.check_frozen_unchanged(
            live.read_log(old_path), live.read_log(new_path), live.new_york_today(now)
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
