from .base import Tokenizer


class BytePairEncoding(Tokenizer):
    def __init__(self) -> None:
        pass

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Creates token vocabulary on a given text.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            verbose (bool): Determines whether training information will be printed.
        """
        pass

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """
        Registers dictionary of special tokens so that the tokenizer can process them individually.
        """
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

    def tokenize(self, text: str) -> list[str]:
        pass