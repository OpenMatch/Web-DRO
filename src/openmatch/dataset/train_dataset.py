# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer

from ..arguments import DataArguments, DRPretrainingDataArguments
from ..data_augmentation_strategy import (Cropping, NullStrategy,
                                          SequentialStrategies)
from ..trainer import DRTrainer

logger = logging.getLogger(__name__)


class TrainDatasetBase:
    '''
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    '''

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self._prepare_data(data_args, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        if not self.is_eval:
            self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(
                    os.path.join(data_args.train_dir, "*.jsonl"))
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset):

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        # self.dataset = self.dataset.shuffle(
        #     seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __len__(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count
    def get_n_clusters(self):
            return self.n_clusters
    def __iter__(self):
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch) if self.trainer.state.epoch else 1
            _hashed_seed = hash(self.trainer.args.seed)
            self.dataset.set_epoch(epoch)
            return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=self.all_columns))
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=self.all_columns))



class MappingTrainDatasetMixin(Dataset):

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        
        sample = self.dataset[0]
        self.all_columns = sample.keys()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(index + self.trainer.args.seed)
            return self.get_process_fn(epoch, _hashed_seed)(group)
        return self.get_process_fn(0, None)(group)


class DRTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            group_positives = example['positives']
            group_negatives = example['negatives']

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_passages.append(self.create_one_example(pos_psg))

            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg))

            assert len(encoded_passages) == self.data_args.train_n_passages

            dict_ = {"query_": encoded_query, "passages": encoded_passages} 
            if 'cluster_id' in example:
                #print("WOW")
                dict_['cluster_id_'] = example['cluster_id']
            return dict_  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, DRTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, DRTrainDataset):
    pass


class DRPretrainDataset(TrainDatasetBase):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DRPretrainingDataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(DRPretrainDataset, self).__init__(tokenizer, data_args, trainer, is_eval, shuffle_seed, cache_dir)
        pretrain_strategies_str = data_args.pretrain_strategies.split(
            ",") if data_args.pretrain_strategies is not None else []
        strategies = []
        for strategy_str in pretrain_strategies_str:
            if strategy_str == "null":
                strategies.append(NullStrategy())
                logger.info("Adding NullStrategy")
            elif strategy_str == "crop":
                strategies.append(Cropping(
                    ratio_min=data_args.cropping_ratio_min, ratio_max=data_args.cropping_ratio_max))
                logger.info("Adding Cropping, ratio_min={}, ratio_max={}".format(
                    data_args.cropping_ratio_min, data_args.cropping_ratio_max))
            else:
                raise ValueError(
                    "Unknown pretraining strategy: {}".format(strategy_str))
        self.apply_strategy = SequentialStrategies(*strategies)

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        text_encoding = self.apply_strategy(text_encoding)
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            content = example[self.data_args.pretrain_target_field]
            encoded_query = self.create_one_example(content, is_query=True)
            encoded_passages = [self.create_one_example(content)]
            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn


class StreamDRPretrainDataset(StreamTrainDatasetMixin, DRPretrainDataset):
    pass
    


class MappingDRPretrainDataset(MappingTrainDatasetMixin, DRPretrainDataset):
    pass


class RRTrainDataset(TrainDatasetBase):

    def create_one_example(self, qry_encoding, psg_encoding) -> BatchEncoding:
        if self.data_args.encode_as_text_pair:
            item = self.tokenizer.encode_plus(
                qry_encoding, psg_encoding,
                truncation='longest_first',
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=True,
            )
        else:
            item = self.tokenizer.encode_plus(
                qry_encoding + psg_encoding,
                truncation='longest_first',
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            group_positives = example['positives']
            group_negatives = example['negatives']

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(
                    hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(
                    hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn


class StreamRRTrainDataset(StreamTrainDatasetMixin, RRTrainDataset):
    pass


class MappingRRTrainDataset(MappingTrainDatasetMixin, RRTrainDataset):
    pass


class QGTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            group_positives = example['positives']
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

            encoded_query = self.create_one_example(qry, is_query=True).input_ids
            encoded_query[encoded_query == self.tokenizer.pad_token_id] == -100
            encoded_psg = self.create_one_example(pos_psg)
            psg_input_ids, psg_attention_mask = encoded_psg.input_ids, encoded_psg.attention_mask
            return {"input_ids": psg_input_ids[0], "attention_mask": psg_attention_mask[0], "labels": encoded_query[0]}

        return process_fn


class StreamQGTrainDataset(StreamTrainDatasetMixin, QGTrainDataset):
    pass


class MappingQGTrainDataset(MappingTrainDatasetMixin, QGTrainDataset):
    pass


class PairwiseDistillationTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = self.create_one_example(example["query"], is_query=True)
            pos = self.create_one_example(example["positive"])
            neg = self.create_one_example(example["negative"])
            score = example["score"]
            return {"query_": qry, "positive_": pos, "negative_": neg, "score_": score}

        return process_fn


class StreamPairwiseDistillationTrainDataset(StreamTrainDatasetMixin, PairwiseDistillationTrainDataset):
    pass


class MappingPairwiseDistillationTrainDataset(MappingTrainDatasetMixin, PairwiseDistillationTrainDataset):
    pass


class ListwiseDistillationTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            passages = example['docs']
            scores = example['scores']
            passages_and_scores = list(zip(passages, scores))

            if len(passages) < self.data_args.train_n_passages:
                if hashed_seed is not None:
                    psgs = random.choices(passages_and_scores, k=self.data_args.train_n_passages)
                else:
                    psgs = [x for x in passages_and_scores]
                    psgs = psgs * 2
                    psgs = psgs[:self.data_args.train_n_passages]
            else:
                _offset = epoch * self.data_args.train_n_passages % len(passages)
                psgs = [x for x in passages_and_scores]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(psgs)
                psgs = psgs * 2
                psgs = psgs[_offset: _offset + self.data_args.train_n_passages]

            for psg in psgs:
                encoded_passages.append(self.create_one_example(psg[0]))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {
                "query_": encoded_query, 
                "passages": encoded_passages,
                "scores_": [x[1] for x in psgs]
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamListwiseDistillationTrainDataset(StreamTrainDatasetMixin, ListwiseDistillationTrainDataset):
    pass


class MappingListwiseDistillationTrainDataset(MappingTrainDatasetMixin, ListwiseDistillationTrainDataset):
    pass


    
'''
{'query_': {'input_ids': [xxx]},'passages': [{'input_ids': [xxx]}*n_passages]}
'''
import os
import json
import torch.distributed
import random
from tqdm import tqdm
import copy
import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import functools

class DROMappingDataset(Dataset):
    pass

'''
class DROMappingDataset(Dataset):
    def preprocess(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def process_one(self, item, N, cluster_id):
        instance = {}
        instance['query_'] = self.preprocess(item['query'], True)
        instance['passages'] = [self.preprocess(item['positives'][0])]
        instance['passages'].extend([self.preprocess(sent) for sent in random.sample(item['negatives'], min(N - 1, len(item['negatives'])))])
        instance['cluster_id'] = cluster_id
        return instance

    def __init__(self, tokenizer, data_args, training_args, shuffle_seed):
        shuffle_seed = 42
        random.seed(shuffle_seed)
        self.data_args = data_args
        self.tokenizer = tokenizer
        
        base_dir = data_args.train_dir
        files = os.listdir(base_dir)
        align_to = training_args.per_device_train_batch_size * torch.distributed.get_world_size()
        #为了保证每个batch的数据来自同一个cluster, 要进行对齐，丢弃不能整除的部分
        
        self.data = {}
        for file in files:
            
            if file != "0" and file != "1":
                continue
            
            tmp = []
            file_path = os.path.join(base_dir, file, "train.jsonl") #这里可能以后要修改
            cluster_id = (file) #这里可能以后要修改
            print(f"Visiting: {file}")
            self.data[cluster_id] = load_dataset("json", data_files=file_path, num_proc=multiprocessing.cpu_count())['train']
            len_ = len(self.data[cluster_id])
            self.data[cluster_id] = self.data[cluster_id].select(range((len_ // align_to) * align_to))
            #self.data[cluster_id] = self.data[cluster_id].select(range(1024))

        
        self.n_cluters = len(self.data.keys())
        self.group_cnt = {key: len(value) for (key, value) in self.data.items()}
        self.map = [] #map: id -> which entry from which group does dataset[id] belongs to 
        self.N = data_args.train_n_passages
        
        if training_args.circular: 
            LEN = min(len(value) for (key, value) in self.data.items())
            counter = 0
            while True:
                for key in self.data:
                    for i in range(align_to):
                        self.map.append((key, counter + i))
                counter += align_to
                if counter == LEN:
                    break
        else:
            ptrs = {clus_id: 0 for clus_id in self.data.keys()}
            safe_set = list(self.data.keys())
            while len(safe_set) != 0:
                key = random.choice(safe_set)
                for i in range(align_to):
                    self.map.append((key, ptrs[key] + i))
                ptrs[key] += align_to
                if ptrs[key] == len(self.data[key]):
                    safe_set.remove(key)
    def __len__(self):
        #return len(self.data)
        return len(self.map)

    def __getitem__(self, idx):
        #return self.data[idx]
        cluster_id, index = self.map[idx]
        #print(f"{idx}, {cluster_id}")
        return self.process_one(self.data[cluster_id][index], self.N, cluster_id)
    
    def get_n_clusters(self):
        return self.n_cluters
'''
    
    
class DROStreamDataset(IterableDataset):
    def preprocess(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item
    def get_process_fn(self, cluster_id):
        def process_one(item, self=self):
            instance = {}
            instance['query_'] = self.preprocess(item['query'], True)
            instance['passages'] = [self.preprocess(item['positives'][0])]
            instance['passages'].extend([self.preprocess(sent) for sent in random.sample(item['negatives'], min(self.N - 1, len(item['negatives'])))])
            instance['cluster_id'] = cluster_id
            return instance
        return process_one

    def get_len(self, file_path):
        count = 0
        with os.popen("wc -l {}".format(file_path)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __len__(self):
        #return sum(self.group_cnt.values())
        if self.total_length > 0:
            return self.total_length
        elif self.truncate:
            return min(self.group_cnt.values()) * self.n_clusters
        else:
            return max(self.group_cnt.values()) * self.n_clusters
    
    def __init__(self, tokenizer, data_args, training_args, shuffle_seed):
        super(DROStreamDataset).__init__()

        shuffle_seed = 42
        random.seed(shuffle_seed)
        self.data_args = data_args
        self.tokenizer = tokenizer
        base_dir = data_args.train_dir
        files = os.listdir(base_dir)
        align_to = training_args.per_device_train_batch_size * torch.distributed.get_world_size()
        self.total_length  = training_args.training_steps * align_to
        self.truncate = training_args.truncate
        #为了保证每个batch的数据来自同一个cluster, 要进行对齐，丢弃不能整除的部分
        
        
        if os.path.exists(os.path.join(base_dir, files[0], "train.jsonl")):
            filename = "train.jsonl"
        elif os.path.exists(os.path.join(base_dir, files[0], "train.hn.jsonl")):
            filename = "train.hn.jsonl"
        else:
            assert False
            
        dataset = load_dataset("json", data_files=os.path.join(base_dir, files[0], filename), streaming=True)["train"]
        sample = list(dataset.take(1))[0]
        self.all_columns = sample.keys()
        
        
        self.data = {}
        self.dataset = {}
        self.iterator = {}
        self.group_cnt = {}
        self.N = data_args.train_n_passages
        for file in files:
            tmp = []
            file_path = os.path.join(base_dir, file, filename) #这里可能以后要修改
            cluster_id = (file) #这里可能以后要修改
            print(f"Visiting: {file}")
            self.dataset[cluster_id] = load_dataset("json", data_files=file_path, streaming=True)["train"]
            #self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=1024) if shuffle_seed is not None else self.dataset
            self.iterator[cluster_id] = iter(self.dataset[cluster_id].map(self.get_process_fn(cluster_id), remove_columns=self.all_columns))
            self.group_cnt[cluster_id] = self.get_len(file_path)
            self.group_cnt[cluster_id] = (self.group_cnt[cluster_id] // align_to) * align_to
            #self.group_cnt[cluster_id] = 16
        
        self.n_clusters = len(self.iterator.keys())
        self.map = [] #map: id -> which entry from which group does dataset[id] belongs to 
        if training_args.circular: 
            counter = 0
            while True:
                for key in self.iterator:
                    for i in range(align_to):
                        self.map.append(key)
                counter += align_to
                if counter >= self.__len__():
                    break
        else:
            assert False
            '''
            ptrs = {clus_id: 0 for clus_id in self.iterator.keys()}
            safe_set = list(self.iterator.keys())
            while len(safe_set) != 0:
                key = random.choice(safe_set)
                for i in range(align_to):
                    self.map.append(key)
                ptrs[key] += align_to
                if ptrs[key] == self.group_cnt[key]:
                    safe_set.remove(key)
            '''
        self.iter_cnt = 0
                    
    def __iter__(self):
        return self #在这里把自己当成一个迭代器，用next方法获取下一个元素
    
    def __next__(self):
        #print(f"Iter {self.iter_cnt}")
        if self.iter_cnt >= self.__len__():
            raise StopIteration
        cluster_id = self.map[self.iter_cnt]
        '''
        if torch.distributed.get_rank() == 0:
            print(cluster_id)
        '''
        self.iter_cnt += 1
        try:
            X = next(self.iterator[cluster_id])
        except StopIteration:
            self.dataset[cluster_id].set_epoch(self.iter_cnt)
            self.iterator[cluster_id] = iter(self.dataset[cluster_id].map(self.get_process_fn(cluster_id), remove_columns=self.all_columns))
            X = next(self.iterator[cluster_id])
        return X
    def get_n_clusters(self):
        return self.n_clusters
    
  

class DROMixedDataset(IterableDataset, DRTrainDataset):
    # def preprocess(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
    #     item = self.tokenizer.encode_plus(
    #         text_encoding,
    #         truncation='only_first',
    #         max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
    #         padding=False,
    #         return_attention_mask=False,
    #         return_token_type_ids=False,
    #     )
    #     return item
    # def get_process_fn(self):
    #     def process_one(item, self=self):
    #         instance = {}
    #         instance['query_'] = self.preprocess(item['query'], True)
    #         instance['passages'] = [self.preprocess(item['positives'][0])]
            
    #         while len(item['negatives']) < self.N - 1:
    #             item['negatives'] = item['negatives'] * 2
            
    #         instance['passages'].extend([self.preprocess(sent) for sent in random.sample(item['negatives'], self.N - 1)])
    #         instance['cluster_id_'] = item['cluster_id']
            
    #         #print(f"Num of negs: {len(item['negatives'])}")
    #         return instance
        
    #     return process_one
    def get_n_clusters(self):
        return self.n_clusters
    
    def __len__(self):
        count = 0
        with os.popen("wc -l {}".format(self.train_path)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count
    
    def __init__(self, tokenizer, data_args, training_args, shuffle_seed):
        super().__init__(tokenizer, data_args)

        shuffle_seed = 42
        random.seed(shuffle_seed)
        self.data_args = data_args
        X = torch.load(training_args.distribution_counter)
        self.n_clusters = X.shape[0]
        self.train_path = data_args.train_path
        self.dataset = load_dataset("json", data_files=self.train_path, streaming=True)["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()
                    
    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(42, 0), remove_columns=self.all_columns))