"""Live database servers for the backend suites.

Session-scoped and lazy: a container starts only when a test asks for it, so a targeted run
(`pytest tests/test_sqlite_registry.py`) starts nothing. A full run pays the start once.

Real servers rather than fakes because the contracts under test are engine behaviour -- the
atomic claim, index-backed ordering, the duplicate-key race -- none of which a fake reproduces.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from testcontainers.community.mongodb import MongoDbContainer
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(scope='session')
def mongo_url() -> Iterator[str]:
    """A connection URL for a live mongod. No database is selected; callers name their own."""
    with MongoDbContainer('mongo:7.0') as container:
        yield container.get_connection_url()


@pytest.fixture(scope='session')
def postgres_dsn() -> Iterator[str]:
    """A libpq connection string for a live Postgres. The database exists; no schema is created."""
    with PostgresContainer('postgres:16-alpine', driver=None) as container:
        yield container.get_connection_url()
