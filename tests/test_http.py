"""Tests du client HTTP, sans réseau : httpx.MockTransport joue le rôle du serveur."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from tresor.config import HTTP_MAX_ATTEMPTS, HTTP_USER_AGENT
from tresor.http import FetchError, build_client, fetch_json, fetch_text, get_with_retries

URL = "https://example.test/data"


class FakeServer:
    """Rejoue une suite de réponses, répète la dernière et note les requêtes reçues."""

    def __init__(
        self, *outcomes: int | Exception | httpx.Response, body: str = "contenu"
    ) -> None:
        self.outcomes = list(outcomes)
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx.Response):
            return outcome
        return httpx.Response(outcome, text=self.body)

    def client(self) -> httpx.Client:
        return build_client(transport=httpx.MockTransport(self))


def test_transient_errors_are_retried_with_exponential_backoff() -> None:
    server = FakeServer(503, 503, 200)
    waits: list[float] = []

    response = get_with_retries(server.client(), URL, sleep=waits.append)

    assert response.status_code == 200
    assert len(server.requests) == 3
    assert waits == [1.0, 2.0]


def test_network_errors_are_retried() -> None:
    server = FakeServer(httpx.ConnectError("coupure"), 200)

    response = get_with_retries(server.client(), URL, sleep=lambda _: None)

    assert response.status_code == 200
    assert len(server.requests) == 2


def test_client_errors_are_not_retried() -> None:
    server = FakeServer(404)

    with pytest.raises(FetchError, match="HTTP 404"):
        get_with_retries(server.client(), URL, sleep=lambda _: None)

    assert len(server.requests) == 1


def test_gives_up_after_max_attempts() -> None:
    server = FakeServer(503)
    waits: list[float] = []

    with pytest.raises(FetchError, match="HTTP 503"):
        get_with_retries(server.client(), URL, sleep=waits.append)

    assert len(server.requests) == HTTP_MAX_ATTEMPTS
    assert len(waits) == HTTP_MAX_ATTEMPTS - 1


def test_failures_never_expose_query_parameters(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    server = FakeServer(503)

    with pytest.raises(FetchError) as caught:
        get_with_retries(
            server.client(), URL, params={"api_key": "SECRET"}, sleep=lambda _: None
        )

    assert "api_key=SECRET" in str(server.requests[0].url)  # la clé part bien vers le serveur
    assert "SECRET" not in str(caught.value)
    assert caught.value.__suppress_context__  # l'erreur d'origine, avec l'URL, est masquée
    assert "SECRET" not in caplog.text


def test_user_agent_is_sent() -> None:
    server = FakeServer(200)

    get_with_retries(server.client(), URL)

    assert server.requests[0].headers["user-agent"] == HTTP_USER_AGENT


def test_cache_avoids_a_second_download(tmp_path: Path) -> None:
    cache_path = tmp_path / "sous_dossier" / "2020.csv"
    server = FakeServer(200)
    client = server.client()

    first = fetch_text(URL, cache_path=cache_path, client=client)
    second = fetch_text(URL, cache_path=cache_path, client=client)

    assert first == second == "contenu"
    assert len(server.requests) == 1
    assert not cache_path.with_name("2020.csv.part").exists()


def test_failed_download_leaves_no_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "2020.csv"

    with pytest.raises(FetchError):
        fetch_text(URL, cache_path=cache_path, client=FakeServer(404).client())

    assert not cache_path.exists()


def test_server_request_to_wait_is_respected() -> None:
    server = FakeServer(httpx.Response(429, headers={"Retry-After": "30"}), 200)
    waits: list[float] = []

    get_with_retries(server.client(), URL, sleep=waits.append)

    assert waits == [30.0]


def test_invalid_address_is_not_retried() -> None:
    server = FakeServer(httpx.UnsupportedProtocol("protocole inconnu"))

    with pytest.raises(FetchError):
        get_with_retries(server.client(), URL, sleep=lambda _: None)

    assert len(server.requests) == 1


def test_key_is_masked_even_if_httpx_logging_is_switched_back_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="httpx")
    server = FakeServer(200)

    get_with_retries(server.client(), URL, params={"api_key": "SECRET"})

    assert "HTTP Request" in caplog.text  # le message de httpx est bien écrit...
    assert "SECRET" not in caplog.text  # ...mais sans la clé
    assert "api_key=***" in caplog.text


def test_content_refused_by_validation_is_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "2020.csv"

    def refuse(text: str) -> None:
        raise ValueError("page d'erreur")

    with pytest.raises(ValueError, match="page d'erreur"):
        fetch_text(URL, cache_path=cache_path, client=FakeServer(200).client(), validate=refuse)

    assert not cache_path.exists()


def test_empty_response_is_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "2020.csv"
    server = FakeServer(200, body="  \n")

    fetch_text(URL, cache_path=cache_path, client=server.client())

    assert not cache_path.exists()


def test_fetch_json_decodes_the_response() -> None:
    server = FakeServer(200, body='{"total": 3}')

    assert fetch_json(URL, client=server.client()) == {"total": 3}
