import logging
from typing import Dict

import numpy as np
import yaml

from models.lightning import LightningTransductiveLearner

logger = logging.getLogger(__name__)


def test_run():
    X_labeled = np.arange(0, 5760).reshape(10, 576)
    X_labeled -= 2000
    y_labeled = np.ones(10)
    X_unlabeled = np.arange(0, 5760).reshape(10, 576)
    with open("./tests/config/training_config.yml", "r") as f:
        training_config: Dict = yaml.load(f, yaml.CLoader)
        batch_size = training_config["batch_size"]
        max_epochs = training_config["max_epochs"]
        transductive_learning_iterartion = training_config["transductive_learning_iterartion"]
        label_data_number = training_config["label_data_number"]

    transductive_learner = LightningTransductiveLearner(
        X_labeled, y_labeled, X_unlabeled, "./tests/config/model_config.yml", batch_size, max_epochs
    )
    X_df, y_df = transductive_learner.run(transductive_learning_iterartion, label_data_number)

    assert X_df.shape == (4, 577)
    assert y_df.shape == (4, 2)
