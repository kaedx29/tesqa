import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers import T5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import numpy as np
from T5Model import *
from T5Dataset import *
from T5DatasetModule import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_MAX_LEN = 512 # Input length
OUT_MAX_LEN = 128 # Output Length
TRAIN_BATCH_SIZE = 8 # Training Batch Size
VALID_BATCH_SIZE = 2 # Validation Batch Size
EPOCHS = 15 # Number of Iteration

nw_json_file_paths = "train-SQuAD-clean.json"

with open(nw_json_file_paths) as f:
    content0 = json.load(f)

df1 = pd.DataFrame(content0["data"])
df1 = df1[0:30000]

for i in range(len(df1)):
  df1['answers'].values[i] = df1['answers'].values[i]["text"][0]

df1 = df1[['context','question', 'answers']]
df1["context"] = df1["context"].str.lower()
df1["question"] = df1["question"].str.lower()
df1["answers"] = df1["answers"].str.lower()

pl.seed_everything(100)
warnings.filterwarnings("ignore")

MODEL_NAME = "muchad/idt5-qa-qg"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length = 512)

print("eos_token: {} and id: {}".format(tokenizer.eos_token, tokenizer.eos_token_id)) # End of token (eos_token)
print("unk_token: {} and id: {}".format(tokenizer.unk_token, tokenizer.eos_token_id)) # Unknown token (unk_token)
print("pad_token: {} and id: {}".format(tokenizer.pad_token, tokenizer.eos_token_id)) # Pad token (pad_token)

def run():

    df_train, df_valid = train_test_split(
        df1[0:30000], test_size=0.2, random_state=101
    )

    df_train = df_train.fillna("none")
    df_valid = df_valid.fillna("none")

    df_train['context'] = df_train['context'].apply(lambda x: " ".join(x.split()))
    df_valid['context'] = df_valid['context'].apply(lambda x: " ".join(x.split()))

    df_train['answers'] = df_train['answers'].apply(lambda x: " ".join(x.split()))
    df_valid['answers'] = df_valid['answers'].apply(lambda x: " ".join(x.split()))

    df_train['question'] = df_train['question'].apply(lambda x: " ".join(x.split()))
    df_valid['question'] = df_valid['question'].apply(lambda x: " ".join(x.split()))


    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    dataModule = T5DatasetModule(df_train, df_valid)
    dataModule.setup()

    device = DEVICE
    models = T5Model()
    models.to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath="working",
        filename="best_checkpoint",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks = checkpoint_callback,
        max_epochs= EPOCHS,
        accelerator="auto"
    )

    trainer.fit(models, dataModule)

run()
