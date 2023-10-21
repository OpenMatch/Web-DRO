# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from openmatch.utils import SimpleTrainPreProcessor as TrainPreProcessor

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--doc_template', type=str, default=None)
parser.add_argument('--query_template', type=str, default=None)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=8)
parser.add_argument('--mp_chunk_size', type=int, default=3000)
parser.add_argument('--shard_size', type=int, default=50000)
parser.add_argument('--with_url', action='store_true')

args = parser.parse_args()


qrel, cluster_id = TrainPreProcessor.read_qrel(args.qrels)

def read_line(l):
    q, nn = l.strip().split('\t')
    nn = nn.split(',')
    random.shuffle(nn)
    # print(cluster_id[q], end=' ')
    # print(type(cluster_id[q]))
    # exit(0)
    if q in cluster_id and cluster_id[q][0] != None:
        return q, qrel[q], nn[:args.n_sample], cluster_id[q][0]
    else:
        return q, qrel[q], nn[:args.n_sample]

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

with open(args.negative_file) as nf:
    pbar = tqdm(map(read_line, nf))
    with Pool(processes=64) as p:
        for x in p.imap(func, pbar, chunksize=args.mp_chunk_size):
            # if x == None:
            #     continue
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:04d}.jsonl'), 'w')
                pbar.set_description(f'split - {shard_id:04d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

# with open(args.negative_file) as nf:
#     pbar = tqdm(map(read_line, nf))
#     counter = 0
#     shard_id = 0
#     f = None

#     for x in pbar:
#         if args.with_url:
#             result = processor.process_one_with_url(x)
#         else:
#             result = processor.process_one(x)
#         if result is None:
#             continue
#         # print(result)
#         # exit(0)
#         counter += 1
#         if f is None:
#             f = open(os.path.join(args.save_to, f'split{shard_id:04d}.jsonl'), 'w')
#             pbar.set_description(f'split - {shard_id:04d}')

#         f.write(result + '\n')
#         if counter == args.shard_size:
#             f.close()
#             f = None
#             shard_id += 1
#             counter = 0

if f is not None:
    f.close()
