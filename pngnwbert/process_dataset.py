import argparse
import os
import glob
import time
import traceback

import nltk
import torch
from pngnwbert.torchmoji.moji import TorchMoji
from pngnwbert.h2p_parser.cmudictext import CMUDictExt
CMUDictExt = CMUDictExt()
try:
    from tqdm import tqdm
except ImportError:
    # create mock tqdm with .write() method
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return self.iterable.__iter__()
        @staticmethod
        def write(*args, **kwargs):
            print(*args, **kwargs)

def find_files(path, ext):
    return glob.glob(os.path.join(path, '**', f'*.{ext}'), recursive=True)

def split_text_into_sentence_chunks(text, min_size=20, max_size=120):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for sentence in sentences:
        # add sentences till 20 chars is reached.
        # if sentence is longer than 120 chars, add it to it's own chunk.
        if len(chunks) == 0:
            chunks.append(sentence)
        elif len(chunks[-1]) + len(sentence) < min_size: # if current chunk is too small, add to it
            chunks[-1] += ' ' + sentence
        elif len(chunks[-1])+len(sentence) > max_size: # if next chunk will be too big, start a new one
            chunks.append(sentence)
        else:
            chunks[-1] += ' ' + sentence
    
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to directory containing plain text files.')
    parser.add_argument('--output', type=str, help='path to dump the processed dataset (text + phonemes + deepmoji latents).')
    args = parser.parse_args()
    print(args)
    
    torchmoji = TorchMoji().cuda()
    
    txt_files = find_files(args.dataset, 'txt')
    
    id = 0
    os.makedirs(args.output, exist_ok=True)
    f = open(os.path.join(args.output, 'dataset.txt'), 'w')
    os.makedirs(os.path.join(args.output, 'torchmoji'), exist_ok=True)
    for txt_file in tqdm(txt_files, smoothing=0.01):
        with open(txt_file, 'r') as txt_f:
            txt = txt_f.read()
        
        # split into chunks between 20 and 120 characters. break on punctuation.
        txt_chunks = split_text_into_sentence_chunks(txt)
        
        torchmoji_inputs = {} # {id: text}
        for txt_chunk in txt_chunks:
            txt_chunk = txt_chunk.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            torchmoji_inputs[id] = txt_chunk[:120*2]
            
            try:
                txt_chunk_arpa = CMUDictExt.convert(txt_chunk.strip())
            except:
                time.sleep(0.1)
                print(f'Failed to convert "{txt_chunk}"')
                traceback.print_exc()
                del torchmoji_inputs[id]
            else:
                # write to file
                f.write(f'{id}\t{txt_chunk}\t{txt_chunk_arpa}\n')
                id += 1
        
        batch_size = 128
        batch = {}
        while len(torchmoji_inputs):
            if len(batch) < batch_size:
                id_i, text = torchmoji_inputs.popitem()
                batch[id_i] = text
            if len(batch) == batch_size or len(torchmoji_inputs) == 0:
                # process batch
                texts = list(batch.values())
                ids = list(batch.keys())
                latents = torchmoji(texts)
                for i, id_i in enumerate(ids):
                    tqdm.write(f'{id_i} {latents[i].shape}')
                    torch.save(
                        latents[i].cpu().data.half().clone(),
                        os.path.join(args.output, 'torchmoji', f'{id_i}.pt')
                    )
                batch = {}
