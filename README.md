## Pipeline for cross domain recommendation using GNNs GAT and SAGE

To run the code, it is needed to download [Amazon](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Amazon.zip) datasetes.
Only `.inter` files are requried for movie and book datasets.

To run the whole pipeline starting from preprocessing step, run `python src/main.py`
Also, steps in the pipeline can run independently -> run `python src/train_gat_neighbor.py`
