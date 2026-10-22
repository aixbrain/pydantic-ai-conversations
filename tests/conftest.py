from pydantic_ai import models

from tests.containers import mongo_url as mongo_url  # re-exported as a fixture
from tests.containers import postgres_dsn as postgres_dsn  # re-exported as a fixture

# No test may hit a real model API.
models.ALLOW_MODEL_REQUESTS = False
