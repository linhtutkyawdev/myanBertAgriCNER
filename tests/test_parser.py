import pytest
from mycner.data.parser import parse_line

def test_normal_record():
    line = "စပါး@CROP|စိုက်ပျိုး@FARM_OP|ရာတွင်@O"
    parsed = parse_line(line, 1)
    assert parsed["tokens"] == ["စပါး", "စိုက်ပျိုး", "ရာတွင်"]
    assert parsed["labels"] == ["CROP", "FARM_OP", "O"]

def test_trailing_pipe():
    line = "စပါး@CROP|စိုက်ပျိုး@FARM_OP|ရာတွင်@O|"
    parsed = parse_line(line, 1)
    assert parsed["tokens"] == ["စပါး", "စိုက်ပျိုး", "ရာတွင်"]
    assert parsed["labels"] == ["CROP", "FARM_OP", "O"]

def test_malformed_missing_separator():
    with pytest.raises(ValueError) as excinfo:
        parse_line("စပါးစိုက်ပျိုး", 1)
    assert "lacks '@' separator" in str(excinfo.value)
