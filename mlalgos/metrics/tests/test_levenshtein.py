import pytest
from mlalgos.metrics import levenshtein_distance


@pytest.mark.parametrize(
    "x, y, distance",
    [
        ("", "", 0),
        ("hello", "", 5),
        ("", "hello", 5),
        ("hello", "hello", 0),
        ("a", "b", 1),
        ("a", "a", 0),
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
    ],
)
def test_levenshtein_distance(x: str, y: str, distance: int) -> None:
    assert levenshtein_distance(x, y) == distance
