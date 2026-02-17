# -*- coding: utf-8 -*-
"""
Запуск препроцессинга: сырые CSV -> предобработанные в pre_datasets/Novosibirsk.
Запускать из корня проекта: python run_preprocess.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.pipeline import run_preprocessing

if __name__ == "__main__":
    saved = run_preprocessing(overwrite=True)
    print(f"Processed {len(saved)} files.")
