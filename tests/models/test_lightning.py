import logging

import numpy as np

from models.lightning import LightningTransductiveLearner

logger = logging.getLogger(__name__)


def test_run():
    X_labeled = np.arange(0, 144000).reshape(10, 14400)
    X_labeled -= 70000
    y_labeled = np.zeros(10)
    X_unlabeled = np.arange(0, 144000).reshape(10, 14400)
    transductive_learner = LightningTransductiveLearner(
        X_labeled, y_labeled, X_unlabeled, "./config/model_config.yml", "./config/training_config.yml"
    )
    X_df, y_df = transductive_learner.run()

    assert X_df.shape == (10, 14401)
    assert y_df.shape == (10, 2)
