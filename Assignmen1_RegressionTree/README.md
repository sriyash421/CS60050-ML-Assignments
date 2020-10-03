### Requirements:

- python3
- numpy     : for data handling
- pandas    : for data i/o
- tqdm      : to show progress bar
- graphviz  : to plot the graph


### To Run :

python -m pip install -r requirements.txt
python main.py


### Latest run results :

- Best mse across different splits: 1.30257e4 Height: 13
- Best max depth:11   Corresponding mse:1.30250e4  Total nodes in tree:18586 
- Nodes removed during pruning:4440  Mse loss after pruning:2.3e4