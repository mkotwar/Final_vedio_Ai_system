from __future__ import annotations

from ..database.client import create_backend_client, health_check
from ..database.config import DatabaseConfig


def main() -> None:
    config = DatabaseConfig.from_env(require_backend_credentials=True)
    client = create_backend_client(config)
    result = health_check(client)
    print(result)


if __name__ == "__main__":
    main()
