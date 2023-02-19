# adapted from https://github.com/yang-zhang/lightning-language-modeling/blob/main/language_model.py
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from transformers.optimization import AdamW
from pngnwbert.transformers_bert.modeling_bert import PnGBert
from pngnwbert.transformers_bert.configuration_bert import BertConfig
from pngnwbert.datamodule import DataModule

class LMTrainer(pl.LightningModule):
    def __init__(self, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()
        self.save_hyperparameters()  # note, all __init__ kwargs are loaded by self.save_hyperparameters(), somehow
        self.model = PnGBert.from_path("checkpoints/last_modified.ckpt") #PnGBert(BertConfig())
    
    def training_step(self, batch, batch_idx):
        model_out = self.model(**batch)
        word_loss = model_out["word_loss"]
        char_loss = model_out["char_loss"]
        moji_loss = model_out["moji_loss"]
        loss = word_loss + char_loss + moji_loss
        self.log('train_loss', loss)
        self.log('train_word_loss', word_loss)
        self.log('train_char_loss', char_loss)
        self.log('train_moji_loss', moji_loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon, )
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=3.5e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--mlm_prob', type=float, default=0.15)
    parser.add_argument('--g2p_prob', type=float, default=0.15)
    parser.add_argument('--p2g_prob', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LMTrainer.add_model_specific_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # ------------
    # data
    # ------------
    data_module = DataModule(
        train_batch_size=args.train_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_kwargs=dict(
            path=args.dataset_path,
            vocab=[v.strip() for v in open(args.vocab_path, 'r', encoding='utf-8').read().splitlines() if v.strip()],
            mlm_prob=args.mlm_prob,
            g2p_prob=args.g2p_prob,
            p2g_prob=args.p2g_prob,
        ),
        padding_id = 0,
    )
    
    # ------------
    # model
    # ------------
    lmmodel = LMTrainer(
        args.learning_rate,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_epsilon,
    )
    
    # ------------
    # checkpoint
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='lm-{epoch:02d}',
        every_n_train_steps=10000,
        save_last=True,
    )
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(lmmodel, data_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    cli_main()