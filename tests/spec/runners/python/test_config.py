"""A case's ``config:``, by the rule every runner applies: an inline map of the Config
settings every binding exposes, an unknown or repeated key refused, never ignored."""

from __future__ import annotations

import pytest
from parser import parse_config


def test_a_config_names_the_settings_the_case_opens_with() -> None:
    assert parse_config("{path_search_budget: 1048576}") == {
        "path_search_budget": 1048576
    }


@pytest.mark.parametrize(
    "text",
    [
        "{path_budget: 1048576}",
        "{path_search_budget: 0}",
        "{path_search_budget: big}",
        "{path_search_budget: +1048576}",
        "{path_search_budget: ١}",
        "{path_search_budget: 1048576, path_search_budget: 2}",
        "path_search_budget: 1048576",
        "{}",
    ],
)
def test_a_config_the_runners_cannot_apply_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_config(text)
