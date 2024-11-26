from T5Dataset import *
import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer

MODEL_NAME = "muchad/idt5-qa-qg"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length = 512)

INPUT_MAX_LEN = 512 # Input length
OUT_MAX_LEN = 128 # Output Length
TRAIN_BATCH_SIZE = 8 # Training Batch Size
VALID_BATCH_SIZE = 2 # Validation Batch Size

class T5DatasetModule(pl.LightningDataModule):
    def __init__(self, df_train, df_valid):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUT_MAX_LEN

    def setup(self, stage=None):
        self.train_dataset = T5Dataset(
        context=self.df_train.context.values,
        question=self.df_train.question.values,
        answers=self.df_train.answers.values
        )

        self.valid_dataset = T5Dataset(
        context=self.df_valid.context.values,
        question=self.df_valid.question.values,
        answers=self.df_valid.answers.values
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
         self.train_dataset,
         batch_size= TRAIN_BATCH_SIZE,
         shuffle=True,
         num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
         self.valid_dataset,
         batch_size= VALID_BATCH_SIZE,
         num_workers=1
        )