# imports
import json
from typing import List

import torch

from pngnwbert.torchmoji.sentence_tokenizer import SentenceTokenizer
from pngnwbert.torchmoji.model_def import torchmoji_feature_encoding
from pngnwbert.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

class TorchMoji:
    def __init__(self, verbose=False):
        if verbose:
            print(f'Tokenizing using dictionary from {VOCAB_PATH}')
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        self.torchmoji_tokenizer = SentenceTokenizer(vocabulary, fixed_length=120)
        
        if verbose:
            print(f'Loading model from {PRETRAINED_PATH}.')
        self.torchmoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
    
    def cuda(self):
        self.torchmoji_model.cuda()
        return self
    
    def cpu(self):
        self.torchmoji_model.cpu()
        return self
    
    def __call__(self, texts: List[str]):
        with torch.no_grad():
            self.torchmoji_tokenizer.fixed_length = max(len(text) for text in texts) * 3
            tokenized, _, _ = self.torchmoji_tokenizer.tokenize_sentences(texts)
            embed = self.torchmoji_model(tokenized)
        return torch.from_numpy(embed)[:, None, :].float()# [B, 1, embed]