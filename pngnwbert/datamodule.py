import pytorch_lightning as pl
from pngnwbert.dataset import Dataset, Collate
from torch.utils.data.dataloader import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, dataloader_num_workers, dataset_kwargs, padding_id):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataset_kwargs = dataset_kwargs
        self.padding_id = padding_id
    
    def setup(self, stage=None):
        self.train_dataset = Dataset(**self.dataset_kwargs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=Collate(self.padding_id),
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
            persistent_workers=True,
            shuffle=True,
        )