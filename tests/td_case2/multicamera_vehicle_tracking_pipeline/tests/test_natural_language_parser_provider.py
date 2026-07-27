from __future__ import annotations

import unittest

from ..api.services.natural_language_search_service import (
    GeminiNaturalLanguageParserProvider,
    NaturalLanguageSearchContext,
    ProviderUnavailableError,
)


class NaturalLanguageParserProviderTests(unittest.TestCase):
    def test_missing_api_key_is_rejected(self) -> None:
        provider = GeminiNaturalLanguageParserProvider(
            api_key="",
            model_name="gemini-2.5-flash",
            timeout_seconds=20,
            max_retries=1,
        )
        with self.assertRaises(ProviderUnavailableError):
            provider.parse_vehicle_search(
                "Find vehicle DL8CBF6268.",
                NaturalLanguageSearchContext(run_code=None, result_scope=None, default_time_tolerance_minutes=15),
            )


if __name__ == "__main__":
    unittest.main()
