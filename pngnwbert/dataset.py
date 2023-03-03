import traceback
import os
import random
from typing import List, Dict, Tuple

import torch
from unidecode import unidecode

from pngnwbert.torchmoji.model_def import torchmoji_emojis
from pngnwbert.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from pngnwbert.torchmoji.global_variables import NB_EMOJI_CLASSES

from pngnwbert.transformers_bert.tokenization_bert import WordpieceTokenizer
from pngnwbert.h2p_parser.cmudictext import CMUDictExt
CMUDictExt = CMUDictExt()

# get_random_line() function that reads a random line from a text file without loading the entire file.
# taken from https://stackoverflow.com/a/56973905 (by user ivan-vinogradov)
def get_random_line(filepath: str) -> str:
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while True:
            pos = random.randint(0, file_size)
            if not pos:  # the first line is chosen
                return f.readline().decode()  # return str
            f.seek(pos)  # seek to random position
            f.readline()  # skip possibly incomplete line
            line = f.readline()  # read next (full) line
            if line:
                return line.decode()
            # else: line is empty -> EOF -> try another position in next iteration

# https://stackoverflow.com/a/3679747
def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

class BadQualityDataError(Exception):
    pass

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, vocab: List[str],
            mlm_prob: float = 0.10, # replace input with mask and predict the original token
            g2p_prob: float = 0.10, # replace phone with mask and use word_id+text to predict the original phone
            p2g_prob: float = 0.10, # replace text with mask and use word_id+phone to predict the original text
            rnd_prob: float = 0.10, # replace input with random token and predict the original token
            word_id_range: Tuple[int, int] = (0, 33147),
            letter_id_range: Tuple[int, int] = (106, 195),
            phone_id_range: Tuple[int, int] = (33148, 33148 + 83),
            ):
        self.path = path # path to dataset.txt file (a tsv with id,text,phoneme)
        self.n_lines = len(open(self.path, 'r').readlines())
        self.n_vocab = len(vocab)
        self.tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='MASKTOKEN')
        torchmoji = torchmoji_emojis(PRETRAINED_PATH)
        self.torchmoji_final_dropout = torchmoji.final_dropout
        self.torchmoji_output_layer = torchmoji.output_layer
        
        self.word_id_range = word_id_range
        self.letter_id_range = letter_id_range
        self.phone_id_range = phone_id_range
        
        self.mlm_prob = mlm_prob # chance the model has to infer a word from context only
        self.g2p_prob = g2p_prob # chance the model has to infer a phoneme from it's grapheme counterpart
        self.p2g_prob = p2g_prob # chance the model has to infer a grapheme from it's phoneme counterpart
        self.rnd_prob = rnd_prob # chance the model has to infer a correct word from a random word
        assert self.mlm_prob + self.g2p_prob + self.p2g_prob + self.rnd_prob <= 1.0, 'mlm_prob + g2p_prob + p2g_prob must be <= 1.0'
    
    def __len__(self):
        return self.n_lines
    
    def process_seq(self, text, phones):
        # ignore text with URLs
        if 'http' in text:
            raise BadQualityDataError('Text contains URL')
        if 'www' in text:
            raise BadQualityDataError('Text contains URL')
        
        # unidecode (and ignore non-english text)
        raw_text_len = len(text)
        text = unidecode(text)
        text_len = len(text)
        if text_len < 0.5 * raw_text_len:
            raise BadQualityDataError('Text contains >50% non-ASCII characters')
        if text_len > 280:
            raise BadQualityDataError(f'Text is too long ({text_len} > 280)\nText: {text}')
        
        max_word_len = max([len(word) for word in text.split()])
        if max_word_len > 63:
            raise BadQualityDataError(f'Text contains a word that is too long ({max_word_len} > 64)')
        
        return self.tokenizer.convert_string_to_model_inputs(text, phones)
    
    def get_masked_inputs(self, seq_word_ids, seq_char_ids, seq_segment_ids, seq_rel_word_pos, seq_bad_arpa):
        n_words = max(seq_rel_word_pos) + 1

        # randomly assign a task for each word
        word_tasks = []
        for i in range(n_words):
            none_p = 1.0 - self.mlm_prob - self.g2p_prob - self.p2g_prob
            task = weighted_choice([
                ('mlm', self.mlm_prob),
                ('g2p', self.g2p_prob),
                ('p2g', self.p2g_prob),
                ('rnd', self.rnd_prob),
                ('copy', none_p/2),
                ('none', none_p/2),
            ])
            word_tasks.append(task)

        seq_word_ids_input = []  # masked input for model
        seq_char_ids_input = []  # masked input for model

        seq_word_ids_target = []  # target for model to predict
        seq_char_ids_target = []  # target for model to predict
        
        def get_rand_word_id():
            return random.randint(*self.word_id_range)
        
        def get_rand_letter_id():
            return random.randint(*self.letter_id_range)
        
        def get_rand_phone_id():
            return random.randint(*self.phone_id_range)
        
        # apply the tasks to the words.
        # (note that each word appears in both the grapheme and phoneme sequences, so both must be masked together for MLM task)
        MASK_ID = self.tokenizer.token_to_id('MASKTOKEN')
        NO_TARGET_ID = -100
        for i in range(len(seq_word_ids)):
            word_id = seq_rel_word_pos[i]
            task = word_tasks[word_id]
            is_letter = seq_segment_ids[i] == 0 # 0 = letter, 1 = phoneme
            is_phone = seq_segment_ids[i] == 1 # 0 = letter, 1 = phoneme
            failed_to_phonemize = seq_bad_arpa[i] # some words are not converted to ARPAbet by the g2p tool. For now, just ignore these words.
            
            if failed_to_phonemize:
                # no masking, no target
                seq_word_ids_input.append(seq_word_ids[i])
                seq_char_ids_input.append(seq_char_ids[i])
                seq_word_ids_target.append(NO_TARGET_ID)
                seq_char_ids_target.append(NO_TARGET_ID)
                continue
            
            if task == 'mlm':
                # mask the word, it's letters and it's phonemes
                seq_word_ids_input.append(MASK_ID)
                seq_char_ids_input.append(MASK_ID)
                seq_word_ids_target.append(seq_word_ids[i])
                seq_char_ids_target.append(seq_char_ids[i])
            elif task == 'g2p' and is_phone:
                # mask the phonemes and leave word+letters as is
                seq_word_ids_input.append(seq_word_ids[i])
                seq_char_ids_input.append(MASK_ID)
                seq_word_ids_target.append(NO_TARGET_ID)
                seq_char_ids_target.append(seq_char_ids[i])
            elif task == 'p2g' and is_letter:
                # mask the letters and leave word+phoneme as is
                seq_word_ids_input.append(seq_word_ids[i])
                seq_char_ids_input.append(MASK_ID)
                seq_word_ids_target.append(NO_TARGET_ID)
                seq_char_ids_target.append(seq_char_ids[i])
            elif task == 'rnd':
                # replace word and letters with random word and letters
                seq_word_ids_input.append(get_rand_word_id())
                if is_letter:
                    seq_char_ids_input.append(get_rand_letter_id())
                elif is_phone:
                    seq_char_ids_input.append(get_rand_phone_id())
                else:
                    raise Exception(f'Invalid segment id {seq_segment_ids[i]}')
                seq_word_ids_target.append(seq_word_ids[i])
                seq_char_ids_target.append(seq_char_ids[i])
            elif task == 'copy':
                # no masking, no target
                seq_word_ids_input.append(seq_word_ids[i])
                seq_char_ids_input.append(seq_char_ids[i])
                seq_word_ids_target.append(seq_word_ids[i])
                seq_char_ids_target.append(seq_char_ids[i])
            else:
                # no masking, no target
                seq_word_ids_input.append(seq_word_ids[i])
                seq_char_ids_input.append(seq_char_ids[i])
                seq_word_ids_target.append(NO_TARGET_ID)
                seq_char_ids_target.append(NO_TARGET_ID)
        
        return seq_word_ids_input, seq_char_ids_input, seq_word_ids_target, seq_char_ids_target
    
    def getitem(self, index):
        with torch.no_grad():
            line = get_random_line(self.path)
            id, text, phones = line.split('\t')
            
            text = unidecode(text).strip()
            text = CMUDictExt.preprocess_text(text)
            phones = CMUDictExt.convert(text)
            
            seq_word_type, seq_word_ids, seq_char_ids, seq_segment_ids, seq_rel_word_pos, seq_rel_char_pos,\
              seq_abs_word_pos, seq_abs_char_pos, seq_subword_pos, seq_bad_arpa = self.process_seq(text, phones)
            
            seq_word_ids_input, seq_char_ids_input, seq_word_ids_target, seq_char_ids_target = \
                self.get_masked_inputs(seq_word_ids, seq_char_ids, seq_segment_ids, seq_rel_word_pos, seq_bad_arpa)
            
            moji_latent = torch.load(os.path.join(os.path.split(self.path)[0], 'torchmoji', f'{id}.pt'))
            moji_latent = self.torchmoji_final_dropout(moji_latent)
            moji_probs = self.torchmoji_output_layer(moji_latent.to(dtype=next(self.torchmoji_output_layer.parameters()).dtype))
            
            return dict(
                id=id,
                
                seq_word_type=seq_word_type,
                seq_word_ids=seq_word_ids,
                seq_char_ids=seq_char_ids,
                
                seq_segment_ids=seq_segment_ids,
                seq_rel_word_pos=seq_rel_word_pos,
                seq_rel_char_pos=seq_rel_char_pos,
                seq_abs_word_pos=seq_abs_word_pos,
                seq_abs_char_pos=seq_abs_char_pos,
                seq_subword_pos=seq_subword_pos,
                
                seq_word_ids_input=seq_word_ids_input,
                seq_char_ids_input=seq_char_ids_input,
                
                seq_word_ids_target=seq_word_ids_target,
                seq_char_ids_target=seq_char_ids_target,
    
                #moji_latent = moji_latent,
                moji_probs = moji_probs,
            )
    
    def __getitem__(self, index, ignore_exception=True):
        for _ in range(20):
            try:
                return self.getitem(index)
            except BadQualityDataError:
                if not ignore_exception:
                    raise
                #traceback.print_exc()
                continue
            except Exception as e:
                if ignore_exception:
                    traceback.print_exc()
                    index = random.randint(0, len(self) - 1)
                else:
                    raise e
        else:
            raise RuntimeError('Failed to get a random item after 10 attempts.')


class Collate: # collate function for DataLoader
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, List[int]]]):
        """Converts a list of sequences into a batch of padded tensor sequences."""
        batch_size = len(batch)
        
        # get the max sequence length
        max_seq_len = max(len(x['seq_char_ids']) for x in batch)
        
        # pad the sequences
        if 1:
            word_type_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            input_word_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            input_char_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            
            segment_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            rel_word_pos_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            rel_char_pos_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            abs_word_pos_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            abs_char_pos_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            subword_pos_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            
            if 'seq_word_ids_input' in batch[0]:
                seq_word_ids_input = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            if 'seq_char_ids_input' in batch[0]:
                seq_char_ids_input = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            
            if 'seq_word_ids_target' in batch[0]:
                seq_word_ids_target = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            if 'seq_char_ids_target' in batch[0]:
                seq_char_ids_target = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            
            if 'moji_latent' in batch[0]:
                seq_moji_target = torch.zeros((batch_size, max_seq_len, 2304), dtype=torch.float)
            
            if 'moji_probs' in batch[0]:
                seq_moji_target = torch.zeros((batch_size, max_seq_len, NB_EMOJI_CLASSES), dtype=torch.float)
        
        for i, d in enumerate(batch):
            word_type_ids[i, :len(d['seq_word_type'])] = torch.tensor(d['seq_word_type'], dtype=torch.long)
            input_word_ids[i, :len(d['seq_word_ids'])] = torch.tensor(d['seq_word_ids'], dtype=torch.long)
            input_char_ids[i, :len(d['seq_char_ids'])] = torch.tensor(d['seq_char_ids'], dtype=torch.long)
            
            segment_ids[i, :len(d['seq_segment_ids'])] = torch.tensor(d['seq_segment_ids'], dtype=torch.long)
            rel_word_pos_ids[i, :len(d['seq_rel_word_pos'])] = torch.tensor(d['seq_rel_word_pos'], dtype=torch.long)
            rel_char_pos_ids[i, :len(d['seq_rel_char_pos'])] = torch.tensor(d['seq_rel_char_pos'], dtype=torch.long)
            abs_word_pos_ids[i, :len(d['seq_abs_word_pos'])] = torch.tensor(d['seq_abs_word_pos'], dtype=torch.long)
            abs_char_pos_ids[i, :len(d['seq_abs_char_pos'])] = torch.tensor(d['seq_abs_char_pos'], dtype=torch.long)
            subword_pos_ids[i, :len(d['seq_subword_pos'])] = torch.tensor(d['seq_subword_pos'], dtype=torch.long)

            if 'seq_word_ids_input' in batch[0]:
                seq_word_ids_input[i, :len(d['seq_word_ids_input'])] = torch.tensor(d['seq_word_ids_input'], dtype=torch.long)
            if 'seq_char_ids_input' in batch[0]:
                seq_char_ids_input[i, :len(d['seq_char_ids_input'])] = torch.tensor(d['seq_char_ids_input'], dtype=torch.long)
            
            if 'seq_word_ids_target' in batch[0]:
                seq_word_ids_target[i, :len(d['seq_word_ids_target'])] = torch.tensor(d['seq_word_ids_target'], dtype=torch.long)
            if 'seq_char_ids_target' in batch[0]:
                seq_char_ids_target[i, :len(d['seq_char_ids_target'])] = torch.tensor(d['seq_char_ids_target'], dtype=torch.long)

            if 'moji_latent' in batch[0]:
                seq_moji_target[i, :len(d['seq_char_ids']), :] = d['moji_latent'] # [1, 2304] is broadcasted to seq len
            
            if 'moji_probs' in batch[0]:
                seq_moji_target[i, :len(d['seq_char_ids']), :] = d['moji_probs'] # [1, 64] is broadcasted to seq len
        
        out_dict = {
            'word_type_ids': word_type_ids,
            'input_word_ids': seq_word_ids_input if 'seq_word_ids_input' in batch[0] else input_word_ids,
            'input_char_ids': seq_char_ids_input if 'seq_char_ids_input' in batch[0] else input_char_ids,
            
            'segment_ids': segment_ids,
            'rel_word_pos_ids': rel_word_pos_ids,
            'rel_char_pos_ids': rel_char_pos_ids,
            'abs_word_pos_ids': abs_word_pos_ids,
            'abs_char_pos_ids': abs_char_pos_ids,
            'subword_pos_ids': subword_pos_ids,
        }
        
        # maybe add loss targets
        if 'seq_word_ids_target' in batch[0]:
            out_dict['seq_word_ids_target'] = seq_word_ids_target
        if 'seq_char_ids_target' in batch[0]:
            out_dict['seq_char_ids_target'] = seq_char_ids_target
        if any(s in batch[0] for s in {'moji_probs', 'moji_latent'}):
            out_dict['seq_moji_target'] = seq_moji_target
        
        return out_dict