from tqdm import tqdm
import pickle
import csv
import torch
from kmeans_pytorch import kmeans
import numpy as np
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
import torch
from collections import Counter
import time
import sys
# with open("/data1/hanpeixuan/OpenMatch/embedding/Anchor_DR/cont_t5_ver3/embeddings.corpus.rank.0.0-3972542", 'rb') as f:
#     data_corpus = pickle.load(f)
#     data_corpus = {int(val): key for key, val in zip(data_corpus[0], data_corpus[1])}

''''''
emb_path = sys.argv[1]
original_path = sys.argv[2]
subdir_name = sys.argv[3]
K = 15000
''''''

#在这里的分组是基于document的



embeddings = []
ids = []

# with open(emb_path, 'rb') as f:
#     data_query = pickle.load(f)
#     data_query = {int(val): key for key, val in zip(data_query[0], data_query[1])}


corpus_files = [file for file in os.listdir(emb_path) if 'corpus' in file]

data_doc = {}
for file in corpus_files:
    with open(os.path.join(emb_path, file), "rb") as f:
        print(file)
        data_ = list(pickle.load(f))
        norms = np.linalg.norm(data_[0], axis=1, keepdims=True)
        data_[0] = data_[0] / norms
        data_ = {int(val): key for key, val in zip(data_[0], data_[1])}
        data_doc.update(data_)

embeddings = [value for key, value in sorted(data_doc.items())]
print(len(embeddings))

with open(os.path.join(original_path, "qrels.train.tsv"), 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    reader = tqdm(reader)
    for row in reader:
        qid, pid = int(row[0]), int(row[2])
        #embeddings.append(data_query[qid])
        ids.append((qid, pid))

#embeddings = embeddings[:1000]

print("START K-Means")

#cluster_indices, cluster_centers = kmeans(torch.tensor(np.array(embeddings)), K, tol = 0.001, device=torch.device('cpu'))
kmeans = MiniBatchKMeans(n_clusters=K, batch_size = 2048, n_init="auto", verbose=1, tol = 0.001, max_iter = 100, max_no_improvement=None).fit(np.array(embeddings))
cluster_indices = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
#print(Counter(cluster_indices))
print("K-Means FINISHED")

group_counter = []
for i in range(K):
    group_counter.append(0)

os.system(f"mkdir -p {os.path.join(original_path, subdir_name)}")
with open(os.path.join(original_path, subdir_name, "qrels.train.tsv"), "w") as file:
    for (q, d) in ids:
        group_counter[cluster_indices[d]] += 1
        file.write(f"{q}\t0\t{d}\t1\t{cluster_indices[d]}\n")

print(group_counter)

pickle.dump(cluster_centers, open(os.path.join(original_path, subdir_name, "centers.pkl"), "wb"))
torch.save(torch.tensor(group_counter), os.path.join(original_path, subdir_name, "counter.pt"))