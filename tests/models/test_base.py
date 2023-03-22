import logging

import numpy as np

from models.base import TransductiveLearner

logger = logging.getLogger(__name__)


class MockTestTransductiveLearner(TransductiveLearner):
    def _training(self, X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray) -> None:
        pass

    def _inference(self, X_unlabeled: np.ndarray) -> np.ndarray:
        return np.sum(X_unlabeled, axis=-1)


def test_run():
    X_labeled = np.arange(0, 5760).reshape(10, 576)
    X_labeled -= 70000
    y_labeled = np.ones(10)
    X_unlabeled = np.arange(0, 5760).reshape(10, 576)
    transductive_learner = MockTestTransductiveLearner(X_labeled, y_labeled, X_unlabeled, "./config/model_config.yml")
    X_df, y_df = transductive_learner.run()

    assert X_df.shape == (10, 577)
    assert y_df.shape == (10, 2)
    assert X_df["id"].values.tolist() == list(reversed(range(10)))
