from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = Path('code/01_retrain_catboost.ipynb')
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=3600,
    kernel_name='python3',
    resources={'metadata': {'path': 'code'}},
    allow_errors=False,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Notebook 1 execution finished and saved.')
