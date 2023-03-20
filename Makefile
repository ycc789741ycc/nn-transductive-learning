test:
	python -m pytest

example:
	python main.py -x data/X_labeled.pkl -y data/y_labeled.pkl -t data/X_unlabeled.pkl -o data/y_model_labeled.pkl
