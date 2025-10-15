import torch
import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering
import numpy as np
import collections
from tqdm.auto import tqdm
import evaluate


# Parameters for extracting answers
n_best = 20
max_answer_length = 30


class BertModule(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=3e-5):
        super().__init__()
        # Load a pre-trained BERT model for question answering
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # Learning rate for the optimizer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # Forward pass: get model outputs
        outputs = self.model(**batch)
        # Extract loss from outputs
        loss = outputs.loss
        # Log the training loss to TensorBoard
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
