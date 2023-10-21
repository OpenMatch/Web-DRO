# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from openmatch.utils import SimpleTrainPreProcessor as TrainPreProcessor


def load_ranking(rank_file, relevance, n_sample, depth, cluster_id):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, _, p_0, _, _, _ = next(lines).strip().split()
        # if isinstance(q_0, str):
        #     q_0 = (int)(q_0)
        # if isinstance(p_0, str):
        #     p_0 = (int)(p_0)
            
        curr_q = q_0
        #print(relevance[0])
        negatives = [] if p_0 in relevance[q_0] else [p_0]
        while True:
            try:
                q, _, p, _, _, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    if (str)(q) in cluster_id and cluster_id[(str)(q)][0] != None:
                        yield (str)(curr_q), relevance[curr_q], negatives[:n_sample], cluster_id[(str)(curr_q)][0]
                    else:
                        yield (str)(curr_q), relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                if (str)(curr_q) in cluster_id and cluster_id[(str)(curr_q)][0] != None:
                    yield (str)(curr_q), relevance[curr_q], negatives[:n_sample], cluster_id[(str)(curr_q)][0]
                else:
                    #print("WOW")
                    yield (str)(curr_q), relevance[curr_q], negatives[:n_sample]
                return


random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--doc_template', type=str, default=None)
parser.add_argument('--query_template', type=str, default=None)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=8)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=3000)
parser.add_argument('--shard_size', type=int, default=50000)
parser.add_argument('--with_url', action='store_true')
args = parser.parse_args()

qrel, cluster_id = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    doc_max_len=args.truncate,
    doc_template=args.doc_template,
    query_template=args.query_template,
    allow_not_found=True
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)
if args.with_url:
    func = processor.process_one_with_url
else:
    func = processor.process_one
    

pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth, cluster_id))
with Pool(processes=64) as p:
    for x in p.imap(func, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:04d}.hn.jsonl'), 'w')
            pbar.set_description(f'split - {shard_id:04d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

# pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth, cluster_id))
# for item in pbar:
#     x = func(item)
#     counter += 1
#     if f is None:
#         f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.jsonl'), 'w')
#         pbar.set_description(f'split - {shard_id:02d}')
#     f.write(x + '\n')

#     if counter == args.shard_size:
#         f.close()
#         f = None
#         shard_id += 1
#         counter = 0

if f is not None:
    f.close()