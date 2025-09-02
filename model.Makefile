# Makefile for training and evaluating image classifier

.PHONY: all generate_metadata train plot clean

CONDA = conda run -p /net/projects2/promega

all: generate_metadata train plot

generate_metadata:
	$(CONDA) python analysis/images/classifier/generate_traning_metadata.py

train: generate_metadata
	$(CONDA) python analysis/images/classifier/train_model_accuracy.py

plot: train
	$(CONDA) python analysis/images/classifier/plot_model_metrics.py

