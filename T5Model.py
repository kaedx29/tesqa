import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW, T5Tokenizer


MODEL_NAME = "muchad/idt5-qa-qg"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length = 512)

class T5Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return output.loss, output.logits


    def training_step(self, batch, batch_idx):

        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["answers"]
        loss, outputs = self(input_ids, attention_mask, labels)


        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["answers"]
        loss, outputs = self(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)