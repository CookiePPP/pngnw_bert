# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Bert."""

# Note - this is a modified version of the original code

import collections
import re

import unicodedata
from typing import List

import torch
import torch.nn.functional as F
from unidecode import unidecode
from pngnwbert.h2p_parser.text.numbers import normalize_numbers
from pngnwbert.h2p_parser.filter import filter_text

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        "bert-base-multilingual-uncased": (
            "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "bert-large-cased-whole-word-masking": (
            "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "bert-base-cased-finetuned-mrpc": (
            "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
        ),
        "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-uncased": (
            "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
        ),
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
        ),
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
        ),
        "wietsedv/bert-base-dutch-cased": (
            "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
        ),
    }
}

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def split_arpa(text):
    # "{DH IH1 S} {W AA1 Z}, {N OW1 T AH0 B AH0 L}."
    # to ["{DH IH1 S}", "{W AA1 Z},", "{N OW1 T AH0 B AH0 L}."]
    split_text = []
    is_in_bracket = 0
    last_index = 0
    for i, char in enumerate(text):
        if char == "{":
            is_in_bracket += 1
        elif char == "}":
            is_in_bracket -= 1
        elif char == " " and not is_in_bracket:
            split_text.append(text[last_index:i].replace('{', '').replace('}', '').strip())
            last_index = i
    split_text.append(text[last_index:].replace('{', '').replace('}', '').strip())
    return split_text
    

def int_interp(t: List, trgt_len: int, mode='nearest') -> torch.Tensor:
    t = torch.tensor(t, dtype=torch.float64)
    return F.interpolate(t.view(1, 1, -1), size=trgt_len, mode=mode).view(-1).long().tolist()


def get_word_type(word: str):
    # 1: lower case, 2: upper case, 3: title case, 4: other
    
    # remove non-alphabetic characters
    word = re.sub('[^a-zA-Z]+', '', word)
    
    if word.islower():
        return 1
    elif word.isupper():
        return 2
    elif word.istitle():
        return 3
    else:
        return 4


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab: List[str], unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
    
    def token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        try:
            return self.vocab.index(token)
        except ValueError:
            return self.vocab.index(self.unk_token)
    
    def id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.vocab[index]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a sequence of tokens (str) in a sequence of ids (int) using the vocab."""
        return [self.token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a sequence of ids (int) in a sequence of tokens (str) using the vocab."""
        return [self.id_to_token(index) for index in ids]
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (str) in a single string."""
        out_string = "".join(tokens).replace("##", "").strip()
        return out_string
    
    def convert_string_to_model_inputs(self, text:str, phones:str, return_dict=False):
        """Converts a string into entire input for a PnGnW BERT model."""
        # normalize text to match with phoneme sequence as closely as possible
        text = normalize_numbers(text)
        text = filter_text(text, preserve_case=True)
        
        # split sequences into words and assign word position IDs
        # "Saying that it was."
        # "{S EY1 IH0 NG} {DH AE1 T} {IH1 T} {W AA1 Z}."
        # becomes
        # {0: {'word': "Saying", 'phones': ["S", "EY1", "IH0", "NG"]},
        text_words = [w.strip() for w in text.strip().split(" ") if w.strip()]
        phone_words = [w.strip() for w in split_arpa(phones.strip()) if w.strip()]
        if len(text_words) != len(phone_words):
            raise NotImplementedError(
                'Number of words in text and phoneme sequences do not match, which is not supported yet.\n'
                f'Text: "{text.strip()}"\n'
                f'Phones: "{phones.strip()}"\n'
                f'\nZipped: {list(zip(text_words, phone_words))}\n\n'
                f'Number of words in text: {len(text_words)}\n'
                f'Number of words in phones: {len(phone_words)}\n')
        words = []
        for i, (text_word, phone_word) in enumerate(zip(text_words, phone_words)):
            # phone_word = '{S EY1 IH0 NG}'
            words.append({
                'word'  : text_word,
                'phones': phone_word.split(" "),
                'arpa_failed': text_word.strip().lower() == phone_word.strip().lower(),
            })

        # Apply wordpiece tokenization to each word
        # "Saying." -> ["Say", "##ing", "."]
        for word in words:
            word['tokens'] = self.tokenize(word['word'])
        
        # Convert tokens to IDs
        # ["Say", "##ing", "."] -> [688, 1013, 39]
        # ["S", "EY1", "IH0", "NG", "."] -> [105, 101, 102, 103, 39]
        # [["S","a","y", " ", "i","n","g", "."] -> [1, 9, 4, 10, 41, 13, 34, 39]
        for word in words:
            word['token_ids'] = self.convert_tokens_to_ids(word['tokens'])
            word['phone_ids'] = self.convert_tokens_to_ids(word['phones'])
            word['letter_ids'] = self.convert_tokens_to_ids(list(word['word']))

        # Merge into single char+phone+word sequence
        # "Saying.[SEP]S EY1 IH0 NG."
        seq_word_type = []
        seq_word_ids = []
        seq_char_ids = []
        seq_segment_ids = []
        seq_rel_word_pos = []
        seq_rel_char_pos = []
        seq_abs_word_pos = []
        seq_abs_char_pos = []
        seq_subword_pos = []
        seq_bad_arpa = []
        cur_rel_word_pos = 0
        cur_rel_char_pos = 0
        cur_abs_word_pos = 0
        cur_abs_char_pos = 0
        # You ever wonder if there's too many variables? Lol
        for segment_id, mode in enumerate(['letter', 'phone']):
            if mode == 'phone':
                cur_rel_word_pos = 0  # reset pos embeds that are relative to segment start
                cur_rel_char_pos = 0  # reset pos embeds that are relative to segment start
    
            for word in words:
                # word.keys() = ['word', 'phones', 'tokens', 'token_ids', 'phone_ids', 'letter_ids']
                char_ids = word['letter_ids'] if mode == 'letter' else word['phone_ids']
                seq_char_ids.extend(char_ids)
        
                word_ids_ = int_interp(word['token_ids'], trgt_len=len(char_ids), mode='nearest')
                seq_word_ids.extend(word_ids_)
        
                word_type = get_word_type(word['word'])  # int
                seq_word_type.extend([word_type] * len(char_ids))
        
                seq_segment_ids.extend([segment_id] * len(char_ids))
                
                seq_bad_arpa.extend([word['arpa_failed'] if mode == 'phone' else False] * len(char_ids))
                
                for i in range(len(char_ids)):
                    seq_rel_word_pos.append(cur_rel_word_pos)
                    seq_rel_char_pos.append(cur_rel_char_pos)
                    seq_abs_word_pos.append(cur_abs_word_pos)
                    seq_abs_char_pos.append(cur_abs_char_pos)
                    seq_subword_pos.append(i)
                    cur_rel_char_pos += 1
                    cur_abs_char_pos += 1
                cur_rel_word_pos += 1
                cur_abs_word_pos += 1
        
        if return_dict:
            return dict(
                seq_word_type=seq_word_type,
                seq_word_ids=seq_word_ids,
                seq_char_ids=seq_char_ids,
                seq_segment_ids=seq_segment_ids,
                seq_rel_word_pos=seq_rel_word_pos,
                seq_rel_char_pos=seq_rel_char_pos,
                seq_abs_word_pos=seq_abs_word_pos,
                seq_abs_char_pos=seq_abs_char_pos,
                seq_subword_pos=seq_subword_pos,
                seq_bad_arpa=seq_bad_arpa)
        else:
            return (
                seq_word_type,
                seq_word_ids,
                seq_char_ids,
                seq_segment_ids,
                seq_rel_word_pos,
                seq_rel_char_pos,
                seq_abs_word_pos,
                seq_abs_char_pos,
                seq_subword_pos,
                seq_bad_arpa)
