import argparse
from typing import Dict, Text, Tuple

import numpy as np
import pandas as pd
import yaml

from models.lightning import LightningTransductiveLearner

MODEL_CONFIG_PATH = "./config/model_config.yml"
TRAINING_CONFIG_PATH = "./config/training_config.yml"
DEFAULT_OUTPUTS_DIR = "./outputs"


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments."""

    parser = argparse.ArgumentParser(description="Transductive Learning for Neural Network.")
    parser.add_argument("-x", type=str, help="X labeled data csv path, columns=X_0, X_1, X_2,...")
    parser.add_argument("-y", type=str, help="Y labeled data csv path, columns=y")
    parser.add_argument("-t", type=str, help="X unlabeled data, columns=X_0, X_1, X_2,...")
    parser.add_argument("-o", type=str, help="Outputs directory.", default=DEFAULT_OUTPUTS_DIR)

    return parser


def get_data_from_parser(arg_parser: argparse.ArgumentParser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get data from parser."""

    X_labeled_path = arg_parser.parse_args().x
    y_labeled_path = arg_parser.parse_args().y
    X_unlabeled_path = arg_parser.parse_args().t

    if X_labeled_path.split(".")[-1] == "csv":
        X_labeled_df = pd.read_csv(X_labeled_path)
    elif X_labeled_path.split(".")[-1] == "pkl":
        X_labeled_df = pd.read_pickle(X_labeled_path)
    elif X_labeled_path.split(".")[-1] == "parquet":
        X_labeled_df = pd.read_parquet(X_labeled_path)
    else:
        raise ValueError("Data file path must be csv or parquet format.")

    if y_labeled_path.split(".")[-1] == "csv":
        y_labeled_df = pd.read_csv(y_labeled_path)
    elif y_labeled_path.split(".")[-1] == "pkl":
        y_labeled_df = pd.read_pickle(y_labeled_path)
    elif y_labeled_path.split(".")[-1] == "parquet":
        y_labeled_df = pd.read_parquet(y_labeled_path)
    else:
        raise ValueError("Data file path must be csv, pickle or parquet format.")

    if X_unlabeled_path.split(".")[-1] == "csv":
        X_unlabeled_df = pd.read_csv(X_unlabeled_path)
    elif X_unlabeled_path.split(".")[-1] == "pkl":
        X_unlabeled_df = pd.read_pickle(X_unlabeled_path)
    elif X_unlabeled_path.split(".")[-1] == "parquet":
        X_unlabeled_df = pd.read_parquet(X_unlabeled_path)
    else:
        raise ValueError("Data file path must be csv or parquet format.")

    X_labeled = X_labeled_df.to_numpy(dtype=np.float32)
    y_labeled = y_labeled_df.to_numpy(dtype=np.float32)[:, 0]
    X_unlabeled = X_unlabeled_df.to_numpy(dtype=np.float32)

    return X_labeled, y_labeled, X_unlabeled


def get_labeled_data_from_transductive_learning(
    X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray, training_config_path: Text, output_dir: Text
) -> Tuple[np.ndarray, np.ndarray]:
    """Get labeled data from transductive learning."""

    with open(training_config_path, "r") as f:
        training_config: Dict = yaml.load(f, yaml.CLoader)
        batch_size = training_config["batch_size"]
        max_epochs = training_config["max_epochs"]
        transductive_learning_iterartion = training_config["transductive_learning_iterartion"]
        label_data_number = training_config["label_data_number"]

    transductive_learner = LightningTransductiveLearner(
        X_labeled, y_labeled, X_unlabeled, MODEL_CONFIG_PATH, batch_size, max_epochs
    )
    X_model_labeled_df, y_model_labeled_df = transductive_learner.run(
        transductive_learning_iterartion, label_data_number
    )
    X_model_labeled_df.reset_index(drop=True).to_pickle(f"{output_dir}/X_model_labeled.pkl")
    y_model_labeled_df.reset_index(drop=True).to_pickle(f"{output_dir}/y_model_labeled.pkl")

    X_model_labeled_df.pop("id")
    y_model_labeled_df.pop("id")
    X_model_labeled = X_model_labeled_df.to_numpy(dtype=np.float32)
    y_model_labeled = y_model_labeled_df.to_numpy(dtype=np.float32)[:, 0]
    return X_model_labeled, y_model_labeled


def t_sne_visualization(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_model_labeled: np.ndarray,
    y_model_unlabeled: np.ndarray,
    output_dir: Text,
) -> None:
    """t-SNE visualization."""

    pass


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    X_labeled, y_labeled, X_unlabeled = get_data_from_parser(arg_parser)
    X_model_labeld, y_model_labeled = get_labeled_data_from_transductive_learning(
        X_labeled, y_labeled, X_unlabeled, TRAINING_CONFIG_PATH, arg_parser.parse_args().o
    )
    t_sne_visualization(X_labeled, y_labeled, X_model_labeld, y_model_labeled, arg_parser.parse_args().o)
