# Demonstration of REIGNN - REcommender Inductive Graph Neural Network | Version 1 

![Pipeline_image](resources/recommender_pipeline_rev4.png#gh-light-mode-only)
![Pipeline_image](resources/recommender_pipeline_rev4dm.png#gh-dark-mode-only)

Welcome to the official repo of the REIGNN demonstration app -- web service for scientific collaborations assessment powered by a GNN-based recommender system. Here we present the source code for the CIKM'22 paper "Demonstration of REIGNN: R&D Team Management with Graph Machine Learning".

 Artyom Sosedka, Anastasia Martynova, Vladislav Tishin, Natalia Semenova, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov)

arXiv PDF: _to be added_.

# Prerequisites

For the REIGNN model:
```
numpy==1.19.5
pandas==1.3.0
torch==1.8.1
torch-sparse==0.6.12  
torch-scatter==2.0.8
torch-cluster==1.5.9
torch-spline-conv==1.2.1
torch-geometric==2.0.3
wandb==0.12.9
```
Attention -- **installing PyTorch Geometric for GPU can be tricky**.

For backend:
```shell
python = "~3.8"
fastapi = "^0.73.0"
pydantic = { extras = ["dotenv"], version = "^1.8.2" }
uvicorn = "^0.17.4"
cookiecutter = "^1.7.3"
loguru = "^0.6.0"
pandas = "^1.4.0"
scipy = "^1.8.0"
networkx = "^2.6.3"
mypy = "^0.931"
ujson = "^5.1.0"
```

For frontend:
```
TBD
```

# Dataset

We propose a unique dataset that could be used for the evaluation of the REIGNN model. The initial data was gathered from the [Semantic Scholar Open Research Corpus](https://api.semanticscholar.org/corpus) and [SCImago Journal & Country Rank website](https://www.scimagojr.com).

<table>
<tr><td>

<table>
  <tr>
    <td colspan="3">CS1021<sub>general</sub></td>
  </tr>
  <tr>
    <td>Network type</td>
    <td>Co-authorship</td>
    <td>Citation</td>
  </tr>
  
  <tr>
    <td>Nodes</td>
    <td>2798829</td>
    <td>2504381</td>
  </tr>
  
  <tr>
    <td>Edges</td>
    <td>30796749</td>
    <td>17934023</td>
  </tr>
 
  <tr>
    <td>Clustering</td>
    <td>0.714</td>
    <td>0.138</td>
  </tr>
</table>
</td><td>

 <table>
  <tr>
    <td colspan="3">CS1021<sub>demo</sub></td>
  </tr>
  <tr>
    <td>Network type</td>
    <td>Co-authorship</td>
    <td>Citation</td>
  </tr>
  
  <tr>
    <td>Nodes</td>
    <td>3808</td>
    <td>12618</td>
  </tr>
  
  <tr>
    <td>Edges</td>
    <td>31332</td>
    <td>34975</td>
  </tr>
 
  <tr>
    <td>Clustering</td>
    <td>0.644</td>
    <td>0.199</td>
  </tr>
</table>
</td></tr> </table>


In order to obtain the full dataset, it is required to download additional files via `download.sh`. The final revision of the file structure includes general and local parts of the dataset.

## General part
We use the large subgraph CS1021<sub>general</sub> extracted from Semantic Scholar Corpus as the basis for the CS1021<sub>demo</sub> dataset. All papers belong to the period from January 1st, 2011 to December 31st, 2021, and related to the area of Computer Science. 

- _SSORC_CS_2010_2021_authors_edge_list.csv_ - common graph edge list.
- _SSORC_CS_2010_2021_authors_edges_papers_indices.csv_ - common table describing relations between edges in a co-authorship graph (collaborations) and nodes in a citation graph (papers).  
- _SSORC_CS_2010_2021_authors_features.csv_ - table with one-hot encoded authors' research interests.
- _SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.csv_ - table with vectorized via Universal Sentence Encoder abstracts of papers.

## Local part
- <...>_authors.edgelist_ - edge list of a dataset citations graph.
- <...>_papers.edgelist_ - edge list of a dataset co-authorship graph.
- <...>_authors_edges_papers_indices.csv_ - table describing relations between edges in a co-authorship graph (collaborations) and nodes in a citation graph (papers). 
- <...>_papers_targets.csv_ - target values for each auxiliary task regarding edges in a co-authorship graph.

# App running

In order to create .venv and install dependencies for the app running, you need to execute `.backend/setup.sh`. For server launch, it is required to use `.backend/start.sh`.

# Model running

The multitask version of the REIGNN model is available at model/REIGNN.py. If you want to perform tests in a separate environment, you can use the following code:

```python
import torch
import torch.nn as nn

from model.dataloader import get_data
from model.utils import run
from model.REIGNN import REIGNN

root_dir = '../'
dataset_name, split_name, split_number = 'CS1021small', '5_0.1', 0
citation_graph, train_data, val_data, test_data, authors_to_papers, batch_list_x, batch_list_owner = get_data(root_dir, dataset_name, split_name, split_number)

# Global
epochs_per_launch, lr = 15000, 0.001
device = 'cuda:0'

# Local
c_conv_num, c_latent_size, a_conv_num, a_latent_size = 2, 128, 3, 384
operator, link_size, heads = "hadamard", 128, 1 

# Multitask weights
mt_weights = [0.05, 0.05, 0.05, 0.05]

# W&B parameters
wandb_output, project_name, entity, group  = False, 'REIGNN', 'test_entity', 'test_group'

# define the model
model = REIGNN(citation_graph.to(device), heads, device,\
                            train_data.to(device), val_data.to(device), test_data.to(device),
                            authors_to_papers,
                            cit_layers = c_conv_num, latent_size_cit = c_latent_size,
                            auth_layers = a_conv_num, latent_size_auth = a_latent_size,
                            link_size = link_size).to(device) 

optimizer, criterion = torch.optim.Adam(model.parameters(), lr=lr), nn.L1Loss()
run(wandb_output, project_name, group, entity, mt_weights, model, optimizer, criterion, operator, batch_list_x, batch_list_owner, epochs_per_launch)

```

# Contact us

If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.

# License

Established code released as open-source software under the MIT license.

# Citation

```
To be added
```

