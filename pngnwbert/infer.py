import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from unidecode import unidecode
from pngnwbert.dataset import Collate

from pngnwbert.transformers_bert.modeling_bert import PnGBert

from pngnwbert.transformers_bert.tokenization_bert import WordpieceTokenizer
from pngnwbert.h2p_parser.cmudictext import CMUDictExt
CMUDictExt = CMUDictExt()

from os.path import abspath, dirname
FILE_DIR = dirname(abspath(__file__))

class PretrainedModel(nn.Module):
    def __init__(self, model_path:str, vocab_path:str = None, scratch_config = None, freeze_model: bool = True):
        super().__init__()
        if vocab_path is None:
            vocab_path = os.path.join(FILE_DIR, 'vocab.txt')
        vocab = [v.strip() for v in open(vocab_path, 'r', encoding='utf-8').read().splitlines() if v.strip()]
        self.tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='MASKTOKEN')
        self.collate = Collate(pad_token_id=0)
        if (model_path is None or not os.path.exists(model_path)):
            self.model = PnGBert(scratch_config)
        else:
            self.model = PnGBert.from_path(model_path, custom_init_kwargs={'config': scratch_config})
            print("Loaded pngbert from", model_path)
        if freeze_model:
            self.freeze_embedding()
            self.freeze_layers()
            self.freeze_head()
        self.model.eval()
        self.out_dim = self.model.config.hidden_size
    
    def freeze_embedding(self, requires_grad=False):
        for p in self.model.bert.embeddings.parameters():
            p.requires_grad = requires_grad
    
    def freeze_layers(self, n_layers:int = None, percent:float = None, requires_grad=False):
        if n_layers is None and percent is not None:
            n_layers = round(len(self.model.bert.encoder.layer) * percent)
        for p in self.model.bert.encoder.layer[:n_layers].parameters():
            p.requires_grad = requires_grad
    
    def freeze_head(self, requires_grad=False):
        for p in self.model.cls.parameters():
            p.requires_grad = requires_grad
    
    def unfreeze_top_layers(self, n_layers:int = None, percent:float = None):
        if n_layers is None and percent is not None:
            n_layers = round(len(self.model.bert.encoder.layer) * percent)
        if n_layers != 0:
            for p in self.model.bert.encoder.layer[-n_layers:].parameters():
                p.requires_grad = True
    
    def forward(self, texts, return_phone_seq=False):
        # process texts and convert to arpa
        texts = [unidecode(text).strip() for text in texts]
        texts = [CMUDictExt.preprocess_text(text) for text in texts]
        phones = [CMUDictExt.convert(text) for text in texts]
        
        # batch, zero-pad and convert to tensor
        seq_dicts = [self.tokenizer.convert_string_to_model_inputs(text, phone, return_dict=True) for text, phone in zip(texts, phones)]
        batch = self.collate(seq_dicts)
        
        # move to model device and run
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        model_out = self.model(**batch)
        
        if return_phone_seq:
            return self.extract_phone_seq(model_out['last_hidden_state'], batch['segment_ids']), model_out
        else:
            return model_out['last_hidden_state'], model_out
    
    def extract_phone_seq(self, x:torch.Tensor, segment_ids:torch.Tensor, phone_id=1):
        """Extract phoneme subsequence from FloatTensor[B, T, D] input and segment_ids"""
        assert x.shape[0] == segment_ids.shape[0], f'Got batch size mismatch: {x.shape[0]} != {segment_ids.shape[0]}'
        assert x.shape[1] == segment_ids.shape[1], f'Got sequence length mismatch: {x.shape[1]} != {segment_ids.shape[1]}'
        mask = segment_ids == phone_id
        lens = mask.sum(dim=1) # [B]
        phone_seq = torch.zeros(x.shape[0], lens.max(), x.shape[2], device=x.device, dtype=x.dtype)
        for i in range(x.shape[0]):
            phone_seq[i, :lens[i]] = x[i, mask[i]]
        return phone_seq