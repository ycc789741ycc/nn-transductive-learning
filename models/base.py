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

    def run(self, iteration: int = ITERATION, label_number: int = LABEL_NUMBER) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run transductive learning to get the most confidence labeled data.

        Args:
            iteration (int, optional): Times of the iteration. Defaults to ITERATION.
            label_number (int, optional): Final return label number. Defaults to LABEL_NUMBER.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: labeled data, labeled data label.
        """

        select_number = label_number // iteration
        X_labeled_ = self.X_labeled.copy()
        y_labeled_ = self.y_labeled.copy()
        unlabeled_mask = np.array([False] * self.X_unlabeled.shape[0])

        all_labeled_indices = np.array([])
        all_model_labeled_X = np.empty((0, *X_labeled_.shape[1:]))
        all_model_labeled_y = np.empty((0, *y_labeled_.shape[1:]))
        for _ in range(iteration):
            X_train, X_valid, y_train, y_valid = train_test_split(X_labeled_, y_labeled_, test_size=0.2)
            self._training(X_train, X_valid, y_train, y_valid)
            model_labeled_indices, model_labeled_X, model_labeled_y = self._get_most_confidence_labeled_data(
                self.X_unlabeled, unlabeled_mask, select_number
            )
            unlabeled_mask[model_labeled_indices] = True
            X_labeled_ = np.append(X_labeled_, model_labeled_X, axis=0)
            y_labeled_ = np.append(y_labeled_, model_labeled_y, axis=0)

            all_labeled_indices = np.append(all_labeled_indices, model_labeled_indices, axis=0)
            all_model_labeled_X = np.append(all_model_labeled_X, model_labeled_X, axis=0)
            all_model_labeled_y = np.append(all_model_labeled_y, model_labeled_y, axis=0)

        all_model_labeled_X_df = pd.DataFrame({"id": all_labeled_indices})
        X_df = pd.DataFrame({f"X_{i}": all_model_labeled_X[:, i] for i in range(all_model_labeled_X.shape[-1])})
        all_model_labeled_X_df = pd.concat([all_model_labeled_X_df, X_df], axis=1)
        all_model_labeled_y_df = pd.DataFrame({"id": all_labeled_indices, "y": all_model_labeled_y})
        return all_model_labeled_X_df, all_model_labeled_y_df

    @abstractmethod
    def _training(self, X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray) -> None:
        """Training model."""

        raise NotImplementedError

    @abstractmethod
    def _inference(self, X_unlabeled: np.ndarray) -> np.ndarray:
        """Inference the X_unlabeled using SVM.

        Returns:
            np.ndarray: Inference result, (-inf, inf).
        """

        raise NotImplementedError

    def _get_most_confidence_labeled_data(
        self, X_unlabeled: np.ndarray, mask: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most confidence model labeled data.

        Args:
            X_unlabeled (np.ndarray): unlabeled data.
            top_k (int): top k confidence labeled data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: model labeled data indices, model labeled data, label.
        """

        outputs: np.ndarray = self._inference(X_unlabeled[~mask])
        outputs_confidence = np.absolute(outputs)
        top_k_confident_slice: list = np.argsort(-outputs_confidence)[:top_k]

        top_k_label: np.ndarray = outputs[top_k_confident_slice]
        top_k_label[top_k_label > 0] = 1
        top_k_label[~(top_k_label > 0)] = 0
        top_k_labeled_data = X_unlabeled[~mask][top_k_confident_slice]

        return np.where(~mask)[0][top_k_confident_slice], top_k_labeled_data, top_k_label
