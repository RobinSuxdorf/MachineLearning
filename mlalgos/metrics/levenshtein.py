def levenshtein_distance(x: str, y: str) -> int:
    """
    Compute the Levenshtein distance between two strings using a recursive approach.

    The Levenshtein distance (or edit distance) is a measure of the number of single-character edits
    (insertions, deletions, or substitutions) required to transform one string into another.

    Args:
        x (str): The first string.
        y (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    if x == "" or y == "":
        return max(len(x), len(y))

    if x[0] == y[0]:
        return levenshtein_distance(x[1:], y[1:])

    return 1 + min(
        levenshtein_distance(x[1:], y),
        levenshtein_distance(x, y[1:]),
        levenshtein_distance(x[1:], y[1:]),
    )
