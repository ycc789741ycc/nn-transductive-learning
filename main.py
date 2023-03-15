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
    parser.add_argument("-x", type=str, help="X labeled data csv path, columns=id, X_0, X_1, X_2,...")
    parser.add_argument("-y", type=str, help="Y labeled data csv path, columns=id, y")
    parser.add_argument("-t", type=str, help="X unlabeled data, columns=id, X_0, X_1, X_2,...")
    parser.add_argument("-o", type=str, help="Outputs directory.", default=DEFAULT_OUTPUTS_DIR)

    return parser


def get_data_from_parser(arg_parser: argparse.ArgumentParser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get data from parser."""

    def check_id(df: pd.DataFrame) -> None:
        if not df["id"].iloc[0] == 0:
            raise ValueError("id is not start from 0.")
        if not df["id"].is_unique:
            raise ValueError("id is not unique.")
        if not df["id"].is_monotonic_decreasing:
            raise ValueError("id is not monotonic decreasing.")
        if not df["id"].iloc[-1] == len(df) - 1:
            raise ValueError("id is not end with len(df) - 1.")

    X_labeled_csv_path = arg_parser.parse_args().x
    y_labeled_csv_path = arg_parser.parse_args().y
    X_unlabeled_csv_path = arg_parser.parse_args().t

    if X_labeled_csv_path.split(".")[-1] == "csv":
        X_labeled_df = pd.read_csv(X_labeled_csv_path)
    elif y_labeled_csv_path.split(".")[-1] == "parquet":
        X_labeled_df = pd.read_parquet(X_labeled_csv_path)
    else:
        raise ValueError("Data file path must be csv or parquet format.")

    if y_labeled_csv_path.split(".")[-1] == "csv":
        y_labeled_df = pd.read_csv(y_labeled_csv_path)
    elif y_labeled_csv_path.split(".")[-1] == "parquet":
        y_labeled_df = pd.read_parquet(y_labeled_csv_path)
    else:
        raise ValueError("Data file path must be csv or parquet format.")

    if X_unlabeled_csv_path.split(".")[-1] == "csv":
        X_unlabeled_df = pd.read_csv(X_unlabeled_csv_path)
    elif y_labeled_csv_path.split(".")[-1] == "parquet":
        X_unlabeled_df = pd.read_parquet(X_unlabeled_csv_path)
    else:
        raise ValueError("Data file path must be csv or parquet format.")

    for df in [X_labeled_df, y_labeled_df, X_unlabeled_df]:
        check_id(df)
    X_labeled = X_labeled_df[X_labeled_df[X_labeled_df.columns != "id"]].to_numpy(dtype=np.float32)
    y_labeled = y_labeled_df[y_labeled_df[y_labeled_df.columns != "id"]].to_numpy(dtype=np.float32)
    X_unlabeled = X_unlabeled_df[X_unlabeled_df[X_unlabeled_df.columns != "id"]].to_numpy(dtype=np.float32)

    return X_labeled, y_labeled, X_unlabeled


def get_labeled_data_from_transductive_learning(
    X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray, training_config_path: Text, output_dir: Text
) -> Tuple[np.ndarray, np.ndarray]:
    """Get labeled data from transductive learning."""

    with open(training_config_path, "r") as f:
        training_config: Dict = yaml.load(f, yaml.CLoader)
        transductive_learning_iterartion = training_config["transductive_learning_iterartion"]
        label_data_number = training_config["label_data_number"]

    transductive_learner = LightningTransductiveLearner(
        X_labeled, y_labeled, X_unlabeled, MODEL_CONFIG_PATH, TRAINING_CONFIG_PATH
    )
    X_model_labeled_df, y_model_labeled_df = transductive_learner.run(
        transductive_learning_iterartion, label_data_number
    )
    X_model_labeled_df.to_csv(f"{output_dir}/X_model_labeled.csv", index=False)
    y_model_labeled_df.to_csv(f"{output_dir}/y_model_labeled.csv", index=False)

    X_model_labeled = X_model_labeled_df[X_model_labeled_df[X_model_labeled_df.columns != "id"]].to_numpy(
        dtype=np.float32
    )
    y_model_labeled = y_model_labeled_df[y_model_labeled_df[y_model_labeled_df.columns != "id"]].to_numpy(
        dtype=np.float32
    )
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
