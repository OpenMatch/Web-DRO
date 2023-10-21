## GroupDRO Dense Retrieval
### Repository for paper: Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs
#### The Code is Based on [OpenMatch](https://github.com/OpenMatch/OpenMatch).

#### Installment
```
git clone git@github.com:Hanpx20/GroupDRO_Dense_Retrieval.git
python setup.py install
cd src/openMatch
pip install -e .
```
Besides the requirements listed above, you also need to install a modified version of [Transformers](https://github.com/Veronicium/AnchorDR/tree/main/transformers) to adapt our [model](https://huggingface.co/OpenMatch/Web-DRO).


#### Embedding Model Training
prerequisites:
```
corpus.tsv(format: d_id title content url)
queries.train.tsv(format: q_id query)
qrels.train.tsv(format: q_id _ d_id 1)
a tokznier of our model
```
According to the paper, training data should be links extracted from the Internet.
Run: `experiments/embedding_model.sh`

After this, you should have an embedding model.
### Data Clustering
prerequisites:
```
corpus.tsv(format: d_id title content)
queries.train.tsv(format: q_id query)
qrels.train.tsv(format: q_id _ d_id 1)
```
According to the paper, training data should be anchor-document pairs.
Run: `experiments/cluster.sh`

After this, you should have a `qrel` file with cluster ids.

### Retrieval Model Training
prerequisites:
```
corpus.tsv(format: d_id title content)
queries.train.tsv(format: q_id query)
[cluster_name]/qrels.train.tsv(format: q_id _ d_id 1 cluster_id)
[cluster_name]/counter.pt(a file to record the size of each cluster)
```
Run: `experiments/DRO_dense_retrieval.sh`
After this, you should have a retrieval model trained with GroupDRO.

### Evaluation
prerequisites: A model to be evaluated

Run: `experiments/eval_marco.sh` for MsMarco
Run: `experiments/eval_beir.sh [model_name]` for BEIR
You need to set certain variables inside these files MANUALLY.

#### Reminders

##### It's recommended to run instructions one by one.
To better organize files, we recommend you to use the following notions:
```
export BASE_DIR=...
export COLLECTION_DIR=$BASE_DIR/data
export PROCESSED_DIR=$BASE_DIR/processed_data
export PLM_DIR=$BASE_DIR/models
export CHECKPOINT_DIR=$BASE_DIR/ckpts
export LOG_DIR=$BASE_DIR/log
export EMBEDDING_DIR=$BASE_DIR/embedding
export RESULT_DIR=$BASE_DIR/res
```
You can refer to [OpenMatch Documentation](https://openmatch.readthedocs.io/en/latest/) for more information.

`openmatch.driver.build_index` and `openmatch.driver.retrieve` can also be accelerated by distribution.

#### Models

You can download our model through [Huggingface Transformers](huggingface.co). 

<b>[Web-DRO(the final model)](https://huggingface.co/OpenMatch/Web-DRO)</b>

<b>[The Embedding Model](https://huggingface.co/OpenMatch/Web-Graph-Embedding)</b> (used for clustering)