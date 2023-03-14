from abc import ABC, abstractmethod
from typing import Text, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ITERATION = 10
LABEL_NUMBER = 1000


class TransductiveLearner(ABC):
    def __init__(
        self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray, model_config_path: Text
    ) -> None:
        with open(model_config_path, "r") as f:
            self.model_config = yaml.load(f, yaml.CLoader)
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        self.model_labeled_indices = np.array([])

    def run(self, iteration: int = ITERATION, label_number: int = LABEL_NUMBER) -> pd.DataFrame:
        """Labeling unlabeled data.

        Returns:
            pd.DataFrame: model labeled indices, model labeled X, model labeled y.
        """
        select_number = label_number // iteration
        X_labeled_ = self.X_labeled
        y_labeled_ = self.y_labeled
        unlabeled_mask = np.array([False] * self.X_unlabeled.shape[0])

        labeled_indices = np.array([])
        labeled_X = np.array([])
        labeled_y = np.array([])
        np.array([])
        for _ in range(iteration):
            X_train, X_valid, y_train, y_valid = train_test_split(X_labeled_, y_labeled_, test_size=0.2)
            self.training(X_train, X_valid, y_train, y_valid)
            model_labeled_indices, model_labeled_X, model_labeled_y = self.get_most_confidence_labeled_data(
                self.X_unlabeled, unlabeled_mask, select_number
            )
            unlabeled_mask[model_labeled_indices] = True
            X_labeled_ = np.concatenate([X_labeled_, model_labeled_X])
            y_labeled_ = np.concatenate([y_labeled_, model_labeled_y])

            labeled_indices = np.concatenate([labeled_indices, model_labeled_indices])
            labeled_X = np.concatenate([labeled_X, model_labeled_X])
            labeled_y = np.concatenate([labeled_y, model_labeled_y])

        return pd.DataFrame({"id": labeled_indices, "X": labeled_X, "y": labeled_y})

    @abstractmethod
    def training(self, X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray) -> None:
        """Training model."""

        raise NotImplementedError

    @abstractmethod
    def inference(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Inference the X_unlabeled.

        Returns:
            np.ndarray: Inference result.
        """

        raise NotImplementedError

    def get_most_confidence_labeled_data(
        self, X_unlabeled: np.ndarray, mask: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most confidence model labeled data.

        Args:
            X_unlabeled (np.ndarray): unlabeled data.
            top_k (int): top k confidence labeled data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: model labeled data indices, model labeled data, label.
        """

        outputs: np.ndarray = self.inference(X_unlabeled[~mask])
        outputs_confidence = np.sum(np.absolute(outputs), axis=1)
        top_k_confident_slice: list = np.argsort(-outputs_confidence)[:top_k]

        top_k_label: np.ndarray = outputs[top_k_confident_slice]
        top_k_label[top_k_label > 0] = 1
        top_k_label[~(top_k_label > 0)] = 0
        top_k_labeled_data = X_unlabeled[top_k_confident_slice]

        return np.where(~mask)[top_k_confident_slice], top_k_labeled_data, top_k_label
