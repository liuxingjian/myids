import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

with open("multi_classification_demo.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
    print("Notebook executed successfully.")
except Exception as e:
    print(f"Error executing notebook: {e}")
