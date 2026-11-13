"""Client HTTP commun : HTTP/1.1 forcé, reprises automatiques et cache disque simple."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from tresor.config import (
    HTTP_BACKOFF_BASE_SECONDS,
    HTTP_MAX_ATTEMPTS,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
)

logger = logging.getLogger(__name__)

SECRET_PARAMETER = re.compile(r"(api_key=)[^&\s\"']+")


class RedactSecrets(logging.Filter):
    """Masque toute clé d'API dans un message de journal avant qu'il soit écrit."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "api_key=" in message:
            record.msg = SECRET_PARAMETER.sub(r"\1***", message)
            record.args = ()
        return True


# Au niveau INFO, httpx écrit l'URL complète, donc aussi la clé FRED : on le fait taire, et le
# filtre masque la clé si une autre configuration (Streamlit, débogage) réactive ces messages.
_httpx_logger = logging.getLogger("httpx")
_httpx_logger.setLevel(logging.WARNING)
_httpx_logger.addFilter(RedactSecrets())

# 429 = trop de requêtes, 5xx = panne côté serveur : ces erreurs sont souvent passagères.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0

QueryParams = Mapping[str, Any]
TextCheck = Callable[[str], None]


class FetchError(RuntimeError):
    """Échec définitif d'un téléchargement.

    Le message ne contient jamais les paramètres de la requête : une clé d'API ne peut donc
    pas se retrouver dans les journaux. Les secrets doivent passer par `params`, pas par `url`.
    """


def build_client(
    transport: httpx.BaseTransport | None = None, timeout: float = HTTP_TIMEOUT_SECONDS
) -> httpx.Client:
    """Crée le client commun. `transport` sert à simuler un serveur dans les tests."""
    return httpx.Client(
        http1=True,
        http2=False,  # certains serveurs du Trésor coupent les flux HTTP/2
        timeout=timeout,
        headers={"User-Agent": HTTP_USER_AGENT},
        follow_redirects=True,
        transport=transport,
    )


def _is_retryable(error: httpx.HTTPError) -> bool:
    """Dit si l'erreur a des chances de disparaître en réessayant."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES
    # Une adresse au protocole invalide ne se corrigera pas en réessayant.
    return isinstance(error, httpx.TransportError) and not isinstance(
        error, httpx.UnsupportedProtocol
    )


def _retry_delay(error: httpx.HTTPError, attempt: int) -> float:
    """Attente exponentielle, allongée si le serveur demande lui-même d'attendre (Retry-After)."""
    delay = HTTP_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = error.response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            delay = max(delay, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
    return delay


def _describe(error: httpx.HTTPError) -> str:
    """Résume l'erreur sans recopier l'URL complète, qui peut contenir une clé."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def get_with_retries(
    client: httpx.Client,
    url: str,
    params: QueryParams | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Fait un GET et réessaie les erreurs passagères en attendant 1 s, 2 s, 4 s, 8 s."""
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            reason = _describe(error)
            if attempt == HTTP_MAX_ATTEMPTS or not _is_retryable(error):
                # « from None » retire l'erreur d'origine, dont le texte contient l'URL complète.
                message = f"Échec du téléchargement de {url} après {attempt} essai(s) : {reason}"
                raise FetchError(message) from None
            delay = _retry_delay(error, attempt)
            logger.warning(
                "Essai %d raté sur %s (%s), reprise dans %.0f s", attempt, url, reason, delay
            )
            sleep(delay)
    raise FetchError(f"Aucun essai effectué sur {url} : HTTP_MAX_ATTEMPTS doit valoir au moins 1")


def _download(url: str, params: QueryParams | None, client: httpx.Client | None) -> httpx.Response:
    """Utilise le client fourni, sinon en ouvre un pour cette seule requête."""
    if client is not None:
        return get_with_retries(client, url, params)
    with build_client() as own_client:
        return get_with_retries(own_client, url, params)


def _write_atomically(path: Path, text: str) -> None:
    """Écrit d'abord un fichier temporaire : un cache à moitié écrit ne sera jamais relu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def fetch_text(
    url: str,
    params: QueryParams | None = None,
    cache_path: Path | None = None,
    client: httpx.Client | None = None,
    validate: TextCheck | None = None,
) -> str:
    """Télécharge un texte. Avec `cache_path`, le réseau n'est appelé qu'une seule fois.

    `validate` lève une erreur si le contenu est anormal. Il passe avant la mise en cache, pour
    qu'une page d'erreur renvoyée avec le statut 200 ne soit jamais gardée.
    """
    if cache_path is not None and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    text = _download(url, params, client).text
    if validate is not None and text.strip():
        validate(text)
    # Une réponse vide peut être un incident passager : la garder créerait un trou définitif.
    if cache_path is not None and text.strip():
        _write_atomically(cache_path, text)
    return text


def fetch_json(
    url: str,
    params: QueryParams | None = None,
    client: httpx.Client | None = None,
) -> Any:
    """Télécharge et décode une réponse JSON."""
    return _download(url, params, client).json()
