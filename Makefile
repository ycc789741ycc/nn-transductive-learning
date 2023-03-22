test:
	python -m pytest

example:
	python main.py -x data/X_labeled.pkl -y data/y_labeled.pkl -t data/X_unlabeled.pkl -o outputs --pca --tsne
