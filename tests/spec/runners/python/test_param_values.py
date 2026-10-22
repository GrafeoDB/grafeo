"""The typing of .gtest parameter values, by the rule of the Rust reference runner.

A .gtest file cannot pass a number past the f64 range (the Rust runner refuses
it when it builds), so this runner checks that case here.
"""

from __future__ import annotations

import pytest
from parser import param_value


def test_a_json_integer_that_fits_64_bits_is_an_int() -> None:
    assert param_value("[3, 5000000000, 9223372036854775807]") == [
        3,
        5000000000,
        9223372036854775807,
    ]
    assert param_value('{"min": -9223372036854775808}') == {"min": -(2**63)}
    assert all(isinstance(value, int) for value in param_value("[-3, 19, 88]"))


def test_a_json_integer_past_64_bits_is_a_float() -> None:
    values = param_value('[9223372036854775808, {"big": -9223372036854775809}]')
    assert values == [9223372036854775808.0, {"big": -9223372036854775809.0}]
    assert isinstance(values[0], float), "2^63 does not fit i64"
    assert isinstance(values[1]["big"], float), "-2^63 - 1 does not fit i64"


def test_a_json_number_past_the_f64_range_is_an_error() -> None:
    for text in ("[1e400]", '{"tiny": [-1e400]}', "[1" + "0" * 400 + "]"):
        with pytest.raises(ValueError, match="out of the f64 range"):
            param_value(text)


def test_a_bare_number_past_the_f64_range_is_an_error() -> None:
    with pytest.raises(ValueError, match="out of the f64 range"):
        param_value("1e999")
