"""Prévision de la qualité des adjudications du Trésor américain."""

from pathlib import Path as _Path

__version__ = "0.1.0"

# Empreinte du code au moment où il est chargé : chemin, date de modification et taille de
# chaque fichier Python du paquet. L'application en ligne la compare à celle des fichiers
# actuels pour relire le code quand le dépôt a changé (voir app/fresh_project.py, qui calcule
# l'empreinte de la même façon).
_source_fingerprint = tuple(
    (str(_file), _file.stat().st_mtime_ns, _file.stat().st_size)
    for _file in sorted(_Path(__file__).resolve().parent.rglob("*.py"))
)
