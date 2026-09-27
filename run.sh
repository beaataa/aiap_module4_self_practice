#!/bin/bash
set -e

python -m pytest tests -q
python src/data_preparation.py
python src/model_training.py
