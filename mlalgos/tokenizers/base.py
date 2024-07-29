from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of integer indices based on the tokenizer's vocabulary.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of integer indices corresponding to the tokens in the input text.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of integer indices into the original text using the tokenizer's vocabulary.

        Args:
            ids (list[int]): A list of integer indices to decode.

        Returns:
            str: The decoded text.
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes a given text into a list of tokens.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str]: A list of tokens extraced from the input text.
        """
        raise NotImplementedError
