import logging
import os
from typing import Dict, Text

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from models.base import ModelConfig, TransductiveLearner

logger = logging.getLogger(__name__)


class LightningTransductiveLearner(TransductiveLearner):
    def __init__(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        model_config_path: Text,
        training_config_path: Text,
    ) -> None:
        super().__init__(X_labeled, y_labeled, X_unlabeled, model_config_path)
        with open(training_config_path, "r") as f:
            training_config: Dict = yaml.load(f, yaml.CLoader)
            self.batch_size = training_config["batch_size"]
            self.max_epochs = training_config["max_epochs"]

    def _training(self, X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray) -> None:
        train_dataloader = DataLoader(
            dataset=TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            dataset=TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid)),
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

        model = LightningNNClassifier(self.model_config)
        checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, monitor="valid_acc", mode="max")
        early_stopping = EarlyStopping(monitor="valid_acc", patience=5, mode="max", verbose=True)
        trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=[checkpoint_callback, early_stopping])
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        self.best_model_path = checkpoint_callback.best_model_path

    def _inference(self, X_unlabeled: np.ndarray) -> np.ndarray:
        model = LightningNNClassifier.load_from_checkpoint(self.best_model_path)
        return model(torch.tensor(X_unlabeled)).detach().numpy()


class LightningNNClassifier(pl.LightningModule):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        for i, layer in enumerate(self.model_config["fully-connected-layers"]):
            setattr(self, f"linear_{i}", nn.Linear(layer["input"], layer["output"]))
            setattr(self, f"dropout_{i}", nn.Dropout(layer["dropout"]))
        self.linear_output = nn.Linear(
            self.model_config["output-layer"]["input"], self.model_config["output-layer"]["output"]
        )
        self.activator = nn.LeakyReLU(0.2)
        self.loss_criterion = torch.nn.BCELoss()

    def accuracy(self, y_pred, y_true):
        y_pred[y_pred >= 0.0] = 1.0
        y_pred[y_pred < 0.0] = -1.0
        res = torch.sum(y_pred == y_true) / (y_true.shape[0] * y_true.shape[1])
        return res

    def forward(self, x: torch.Tensor):
        for i in range(len(self.model_config["fully-connected-layers"])):
            x = getattr(self, f"linear_{i}")(x)
            x = getattr(self, f"dropout_{i}")(x)
            x = self.activator(x)
        x = self.linear_output(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_criterion(logits, y)
        self.log("valid_loss", loss)

        return {"logits": logits, "ground_truths": y}

    def validation_epoch_end(self, outputs: Dict[Text, torch.Tensor]):
        logits, ground_truths = [], []
        for output in outputs:
            output: torch.Tensor
            logits.extend(output["logits"].tolist())
            ground_truths.extend(output["ground_truths"].tolist())
        evaluation = self.accuracy(torch.tensor(logits), torch.tensor(ground_truths))
        self.log("valid_acc", evaluation)
        logger.info(f"validation accuracy: {evaluation}")

    def test_step(self, test_batch: torch.Tensor, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.accuracy(logits, y)
        self.log("test_loss", loss)

        return {"logits": logits, "ground_truths": y}

    def test_epoch_end(self, outputs: Dict[Text, torch.Tensor]):
        logits, ground_truths = [], []
        for output in outputs:
            output: torch.Tensor
            logits.extend(output["logits"].tolist())
            ground_truths.extend(output["ground_truths"].tolist())
        evaluation = self.accuracy(torch.tensor(logits), torch.tensor(ground_truths))
        logger.info(f"testing accuracy: {evaluation}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
