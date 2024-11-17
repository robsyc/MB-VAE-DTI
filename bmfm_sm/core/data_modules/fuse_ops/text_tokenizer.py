import collections
import importlib
import os

import regex as re
from transformers import BertTokenizer

# these need to be added to the vocab
unk_token = "[UNK]"
sep_token = "[SEP]"
pad_token = "[PAD]"
cls_token = "[CLS]"
mask_token = "[MASK]"


class TextTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = "resources/vocab.txt", **kwargs):
        if vocab_file == "resources/vocab.txt":
            vocab_file = importlib.resources.files("bmfm_sm.resources").joinpath(
                "vocab.txt"
            )
        super().__init__(vocab_file, **kwargs)
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")
        self.pattern = "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.regex_tokenizer = re.compile(self.pattern)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.cls_token] + token_ids_0 + [self.sep_token]
        cls = [self.cls_token]
        sep = [self.sep_token]
        return cls + token_ids_0 + sep + token_ids_1 + sep


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
