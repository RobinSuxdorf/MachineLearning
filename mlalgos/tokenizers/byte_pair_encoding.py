from typing import Optional, TypeAlias
import regex as re
from .base import Tokenizer

PairDict: TypeAlias = dict[tuple[int, int], int]

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BytePairEncoding(Tokenizer):
    def __init__(self):
        super().__init__()
        self._compiled_gpt_pattern = re.compile(GPT4_SPLIT_PATTERN)
        self._special_tokens: dict[str, int] = {}
        self._inverse_special_tokens: dict[int, str] = {}
        self._vocab: dict[int, bytes] = {idx: bytes({idx}) for idx in range(256)}
        self._merges: PairDict = {}

        _special_token_pattern = (
            f"({'|'.join(re.escape(tok) for tok in self._special_tokens)})"
        )
        self._compiled_special_token_pattern = re.compile(_special_token_pattern)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Creates token vocabulary on a given text.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            verbose (bool): Determines whether training information will be printed.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self._compiled_gpt_pattern, text)

        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: PairDict = {}
        vocab = {idx: bytes({idx}) for idx in range(256)}

        for i in range(num_merges):
            stats: PairDict = {}
            for chunk_ids in ids:
                self._get_stats(chunk_ids, stats)

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"Merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurences"
                )

        self._merges = merges
        self._vocab = vocab

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """
        Registers dictionary of special tokens so that the tokenizer can process them individually.

        Args:
            special_tokens (dict[str, int]): Dictionary containing the special tokens as key.
        """
        self._special_tokens = special_tokens
        self._inverse_special_tokens = {v: k for k, v in special_tokens.items()}

        _special_token_pattern = (
            f"({'|'.join(re.escape(tok) for tok in special_tokens)})"
        )
        self._compiled_special_token_pattern = re.compile(_special_token_pattern)

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """
        Returns the token ids of the chunk.

        Args:
            chunk_bytes (bytes): Byte representation of the chunk.

        Returns:
            list[int]: List of token ids.
        """
        ids = list(chunk_bytes)

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self._merges.get(p, float("inf")))

            if pair not in self._merges:
                break

            idx = self._merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """
        Returns the token ids of the text.

        Args:
            text (str): The string to be encoded.

        Returns:
            list[int]: List of token ids.
        """
        text_chunks = re.findall(self._compiled_gpt_pattern, text)

        ids: list[int] = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text: str) -> list[int]:
        if not self._special_tokens:
            return self._encode_ordinary(text)

        special_chunks = re.split(self._compiled_special_token_pattern, text)

        ids: list[int] = []
        for chunk in special_chunks:
            if chunk in self._special_tokens:
                ids.append(self._special_tokens[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes: list[int] = []
        for idx in ids:
            if idx in self._vocab:
                text_bytes.append(self._vocab[idx])
            elif idx in self._inverse_special_tokens:
                text_bytes.append(self._inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")

        text_bytes = b"".join(text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def tokenize(self, text: str) -> list[str]:
        ids = self.encode(text)
        tokens: list[str] = []

        for idx in ids:
            if idx in self._vocab:
                tokens.append(self._vocab[idx].decode("utf-8", errors="replace"))
            elif idx in self._inverse_special_tokens:
                tokens.append(self._inverse_special_tokens[idx])
            else:
                raise ValueError(f"Invalid token id: {idx}")
        return tokens

    def _get_stats(self, ids: list[int], counts: Optional[PairDict] = None) -> PairDict:
        """
        Counts the occurences of consecutive tokens in ids.

        Args:
            ids: (list[int]): The ids of the tokens.
            counts (Optional[PairDict]): Optional counts dictionary, which will be used when text is previously chunked.

        Returns:
            PairDict: Dictionary containing the counts for each consecutive pair of tokens.
        """
        counts = {} if counts is None else counts

        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """
        Replace all consecutive occurences of pair with the new token idx.

        Args:
            ids (list[int]): The ids of the tokens.
            pair (tuple[int, int]): The pair which will be merged.
            idx (int): The id of the new token.

        Returns:
            list[int]: Returns a list where all occurences of pair will be replaced with idx.
        """
        new_ids: list[int] = []
        i = 0

        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids
