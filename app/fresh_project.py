"""Relecture du code du projet quand il a changé, pour l'application en ligne.

Quand le dépôt change, Streamlit relit les fichiers du dossier `app/`, mais pas les modules de
`src/tresor/` déjà chargés : sans ce module, l'application en ligne continuerait à tourner avec
l'ancien code jusqu'à un redémarrage manuel. Ici, si un fichier Python du projet a changé
depuis son chargement, tous les modules du projet sont oubliés puis relus d'un coup (jamais un
mélange d'ancien et de nouveau code).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

Fingerprint = tuple[tuple[str, int, int], ...]


def source_fingerprint(folder: Path) -> Fingerprint:
    """Chemin, date de modification et taille de chaque fichier Python du dossier : le même
    calcul que celui que fait le paquet `tresor` quand il est chargé."""
    files = sorted(folder.resolve().rglob("*.py"))
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)


def load_fresh(package: str, source: Path) -> ModuleType:
    """Le paquet `package` (dans le dossier `source`), relu si un de ses fichiers a changé.

    Le paquet note son empreinte quand il est chargé (`_source_fingerprint`). Un paquet chargé
    avec le code actuel n'est jamais relu ; un paquet chargé avant une modification, ou par
    une version qui ne notait pas d'empreinte, est oublié avec tous ses modules, puis relu.
    """
    loaded = sys.modules.get(package)
    current = source_fingerprint(source / package)
    if loaded is not None and getattr(loaded, "_source_fingerprint", None) != current:
        stale = [name for name in sys.modules if name == package or name.startswith(f"{package}.")]
        for name in stale:
            sys.modules.pop(name, None)
    return importlib.import_module(package)
