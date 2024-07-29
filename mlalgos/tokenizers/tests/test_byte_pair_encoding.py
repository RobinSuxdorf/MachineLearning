import os
import pytest
from mlalgos.tokenizers import BytePairEncoding


def _unpack(text: str) -> str:
    """
    Function for unpacking text from other files. We use this because pytest print the arguments to the console.

    Args:
        text (str): The text which will be unpacked.

    Returns:
        str: The unpacked text.
    """
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(dirname, text[5:])
        content = open(file, "r", encoding="utf-8").read()
        return content
    return text


@pytest.mark.parametrize("text", ["", "?", "Hello World!", "FILE:obiwankenobi.txt"])
def test_encode_decode_identity(text: str) -> None:
    text = _unpack(text)
    tokenizer = BytePairEncoding()

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert text == decoded


def test_special_tokens_processing() -> None:
    tokenizer = BytePairEncoding()

    tokenizer.register_special_tokens({"<|startoftext|>": 1000, "<|endoftext|>": 1001})

    encoded = tokenizer.encode("<|startoftext|>Hello World!<|endoftext|>")

    assert encoded == [
        1000,
        72,
        101,
        108,
        108,
        111,
        32,
        87,
        111,
        114,
        108,
        100,
        33,
        1001,
    ]


def test_wikipedia_example() -> None:
    """
    https://en.wikipedia.org/wiki/Byte_pair_encoding
    """
    test_string = "aaabdaaabac"

    tokenizer = BytePairEncoding()
    tokenizer.train(test_string, 256 + 3)

    tokens = tokenizer.tokenize(test_string)

    assert tokens == ["aaab", "d", "aaab", "a", "c"]
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string
