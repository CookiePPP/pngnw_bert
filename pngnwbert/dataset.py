import random
import traceback
from typing import List

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], rank: int = 0, n_rank: int = 1):
        self.paths = paths
        self.rank = rank
        self.n_rank = n_rank
    
    def __len__(self):
        return len(self.wd['wav_path'])
    
    def getitem(self, index):
        with torch.no_grad():
            wd_i = index_wd(self.wd, index)  # select one sample
            for key in self.cached_keys:
                if key not in wd_i:
                    wd_i = self.cachers[key].try_load(wd_i)
            wd_i = self.pipeline(keys=self.keys, wd=wd_i, remove_intermediates=False)  # process the sample
            assert all(v.isfinite().all() for v in wd_i.values() if torch.is_tensor(v)), 'NaN or Inf found in wd_i'
        return wd_i  # return the processed sample for the model(s) to use
    
    def __getitem__(self, index, ignore_exception=True):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                if ignore_exception:
                    traceback.print_exc()
                    print(f'Path = "{self.wd["wav_path"][index]}"')
                    index = random.randint(0, len(self) - 1)
                else:
                    raise e
        else:
            raise RuntimeError('Failed to get a random item after 10 attempts.')
