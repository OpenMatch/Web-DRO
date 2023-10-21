# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union
import wandb
import datasets
import torch
from math import isinf, isnan
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from transformers.file_utils import is_datasets_available
from transformers.trainer import Trainer, TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data.distributed import DistributedSampler
from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss
from transformers.utils import is_sagemaker_mp_enabled
import psutil
import GPUtil
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
    
def calc_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
        return 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters])
        return torch.norm(total_norm, 2.0).item()

class DRTrainer(Trainer):
    def __init__(self, delta_model=None, *args, **kwargs):
        super(DRTrainer, self).__init__(*args, **kwargs)
        #self.args = args
        self.delta_model = delta_model
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        self.wandb_id = kwargs['args'].wandb_id
        self.log_grad_norm = kwargs['args'].log_grad_norm
        self.log_emb_grad = kwargs['args'].log_emb_grad
        self.output_dir =  kwargs['args'].output_dir
        if self.log_grad_norm or self.log_emb_grad:
            self.grad_list = []
            self.emb_grad_list = []
        if torch.distributed.get_rank() == 0:
            print(self._dist_loss_scale_factor)
            
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.delta_model:
            logger.info("Saving delta model to %s", output_dir + "/delta_model")
            self.delta_model.save_finetuned(output_dir + "/delta_model")

        if self.log_emb_grad:
            try:
                X = torch.load(os.path.join(self.output_dir, "emb_grad.pt"))
            except:
                X = torch.tensor([]).to(torch.device("cpu"))
            if len(self.emb_grad_list) != 0:
                X = torch.cat((X, torch.stack(self.emb_grad_list).flatten(0, 1)), 0)
            torch.save(X, os.path.join(self.output_dir, "emb_grad.pt"))
            self.emb_grad_list = []
        
    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared
        
        '''
        prepared = [Q, D]
        Q = {'input_ids": [bsz,len], 'attention_mask': [bsz,len]}
        D = {'input_ids": [bsz*n_passages,len], 'attention_mask': [bsz*n_passages,len]}
        '''
        

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.args.use_dro:
            #train_sampler = SequentialSampler(train_dataset)
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
        else:
            train_sampler = self._get_train_sampler()
            
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.distillation:
            if self.args.distil_mode == "pairwise":
                query, positive, negative, score = inputs
                outputs = model(query=query, positive=positive, negative=negative, score=score)
            else:  # listwise
                query, passage, score = inputs
                outputs = model(query=query, passage=passage, score=score)
        else:
            query, passage = inputs
            outputs = model(query=query, passage=passage, log_emb_grad = self.log_emb_grad)
            
        if self.log_emb_grad:
            return (outputs.loss, outputs, outputs.q_reps) if return_outputs else (outputs.loss, outputs.q_reps)
        else:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.log_emb_grad:
            q_reps = loss[1]
            loss = loss[0]
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        if self.log_grad_norm:
            x = calc_grad_norm(model)
            if not (isinf(x) or isnan(x)):
                self.grad_list.append(x)
                
        if self.log_emb_grad:
            # print(q_reps.grad)
            # exit(0)
            self.emb_grad_list.append(q_reps.grad.to(torch.device('cpu')))
            q_reps.grad = None
            del q_reps
        
        return loss.detach() / self.args.gradient_accumulation_steps / self._dist_loss_scale_factor
    
    # def training_step(self, *args):
    #     x = super(DRTrainer, self).training_step(*args)
    #     return x / self._dist_loss_scale_factor
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        print("Start Training!!!")
        if self.wandb_id != "" and torch.distributed.get_rank() == 0:
            wandb.init(project="ClueWeb_Final", id=self.wandb_id)
        super().train(resume_from_checkpoint, trial, **kwargs)
        # if self.log_grad_norm and torch.distributed.get_rank() == 0:
        #     torch.save(self.grad_list, os.path.join(self.output_dir, "gradnorm.pt"))
        if self.wandb_id != "" and torch.distributed.get_rank() == 0:
            wandb.finish()
            
    def log(self, logs: Dict[str, float]):
        logs['cpu_mem(GB)'] = psutil.virtual_memory().used / (1024 ** 3)
        logs['gpu_mem(GB)'] = GPUtil.getGPUs()[0].memoryUsed / 1024
        if self.log_grad_norm:
            if len(self.grad_list) == 0:
                logs['grad_norm'] = 0
            else:
                logs['grad_norm'] = sum(self.grad_list) / len(self.grad_list)
            self.grad_list = []
        super().log(logs)

    # def save_model(self, output_dir=None):
    #     # 自定义保存逻辑
    #     # ...
    #     print("B")
    #     super().save_model(output_dir)



def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCDenseTrainer(DRTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCDenseTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor



#print(self.lr_scheduler.get_last_lr()[0])
def as_tsv(x, N):
    x = ["{:.{}f}".format(i * N, 5) for i in x.tolist()]
    x = x[:5000]
    str_ = ""
    for i in x[:-1]:
        str_ += (str)(i) + '\t'
    str_ += (str)(x[-1]) + '\n'
    return str_
    

class DROTrainer(DRTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing DRO Trainer')
        os.system(f"mkdir {kwargs['args'].output_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.n_clusters = kwargs['train_dataset'].get_n_clusters()
        self.train_alpha = kwargs['args'].train_alpha
        
        if self.train_alpha:
            if kwargs['reference_model']:
                self.reference_model = kwargs['reference_model'].to(device)
                self.reference_model.eval()
            else:
                self.reference_model = None
            if 'init_alpha' in kwargs and kwargs['init_alpha'] != None:
                self.alpha = kwargs['init_alpha'].to(device)
                self.alpha.requires_grad_(False)
                if torch.distributed.get_rank() == 0:
                    print(self.alpha)
            else:
                self.alpha = torch.ones(self.n_clusters, requires_grad=False).to(device) / self.n_clusters
            self.alpha_list = []
            self.relative_alpha_lr = kwargs['args'].alpha_lr / kwargs['args'].learning_rate
            self.mom = kwargs['args'].momentum
            self.alpha_save_path = kwargs['args'].output_dir + "/alpha.txt"
            # with open(self.alpha_save_path, "w"):
                # pass
        else:
            self.reference_model = None
            self.alpha = kwargs['init_alpha'].to(device)
            
        if 'reference_model' in kwargs:
            del kwargs['reference_model']
        if 'init_alpha' in kwargs:
            del kwargs['init_alpha']

        super(DROTrainer, self).__init__(*args, **kwargs)

        #记录rate是为了适配主LR的scheduler
        #self.group_cnt = kwargs['train_dataset'].group_cnt
        self.adjust = kwargs['args'].adjust_on_group_size
        self.output_dir = kwargs['args'].output_dir
        self.interval = kwargs['args'].alpha_log_interval
        self.save_steps = kwargs['args'].save_steps
        self.mixed_dataset = kwargs['args'].mixed_dataset
        self.smooth_factor = kwargs['args'].smooth_factor
        self.freeze_last_group = kwargs['args'].freeze_last_group
        if kwargs['args'].distribution_counter != None:
            self.counter_weight = torch.load(kwargs['args'].distribution_counter)
            self.counter_weight.requires_grad_(False)
            if self.freeze_last_group:
                self.counter_weight = torch.mean(self.counter_weight[:-1].to(torch.float)) / self.counter_weight[:-1]
                self.counter_weight = torch.cat((self.counter_weight, torch.tensor([1], requires_grad=False)))
            else:
                self.counter_weight = torch.mean(self.counter_weight.to(torch.float)) / self.counter_weight
            '''
            条件1 每组的weight * cnt都相等
            条件2 当各组cnt一致时各组的weight都为1
            故 weight=平均每组元素数/本组元素数
            eg. 三组分别有5, 7, 8个元素
            则weight分别为1.33, 0.95, 0.83
            10, 1, 1
            0.4, 4, 4

            '''
        else:
            self.counter_weight = torch.ones(self.n_clusters, requires_grad=False)
        self.counter_weight = self.counter_weight.to(device)
        
        self.alpha_grad_cache = torch.zeros(self.n_clusters, requires_grad=False).to(device)
        self.multiplier = torch.zeros(self.n_clusters, requires_grad=False).to(device)
        
        self.wandb_id = kwargs['args'].wandb_id
        
            
        
    def compute_loss(self, model, inputs, return_outputs=False):
        query, passage, cluster_id = inputs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #这一步通过cross device得到所有样本的cluster id
        cluster_id = torch.tensor([int(i) for i in cluster_id], requires_grad=False).to(device)
        outputs = model(query=query, passage=passage, cluster_id=cluster_id)
        cluster_id = outputs.cluster_id
        # if  torch.distributed.get_rank() == 0:
        #     print(cluster_id)
        #在有参考模型的情况下计算excess loss
        if self.train_alpha and self.reference_model:
            ref_output = self.reference_model(query=query, passage=passage)
            outputs.loss -= ref_output.loss.detach()
            outputs.loss_detail -= ref_output.loss_detail.detach()
        #self.counter += 1

        if self.train_alpha:
            #权重的学习率和主学习率同频变化
            curr_alpha_lr = self.relative_alpha_lr * self.lr_scheduler.get_last_lr()[0]
            
            if self.mixed_dataset:
                #这一步计算batch里来自各个组别的样本的loss之和
                tmp = torch.bincount(cluster_id, weights=outputs.loss_detail.detach(), minlength=self.n_clusters)              #tmp.requires_grad_(False)
                #乘counter_weight是为了平衡不同组别的loss，除batch size是为了和另一种情况统一起来
                tmp = tmp * self.counter_weight / cluster_id.shape[0]
                self.alpha_grad_cache += tmp
                del tmp
                #对权重梯度使用cache，更新频率和Alpha记录的频率一致
                if (self.state.global_step + 1) % self.interval == 0:
                    curr_grad = self.alpha_grad_cache
                    self.alpha_grad_cache = torch.zeros(self.n_clusters, requires_grad=False).to(device)
                else:
                    curr_grad = torch.zeros(self.n_clusters, requires_grad=False).to(device)
            else:
                #此设定下每个batch都来自同一组，直接乘即可
                curr_grad = torch.zeros(self.n_clusters, requires_grad=False)
                curr_grad[cluster_id[0]] = outputs.loss
                
            #处理Momentum
            self.multiplier = self.mom * self.multiplier + (1 - self.mom) * curr_grad.to(device) #Momentum
            del curr_grad

            # if (self.state.global_step + 1) % self.interval == 0 and torch.distributed.get_rank() == 0:
            #     print(self.multiplier[:10])
            #更新并归一化权重
            if self.freeze_last_group:
                self.alpha[:-1] = self.alpha[:-1] * torch.exp(curr_alpha_lr * self.multiplier[:-1])
                self.alpha[:-1] = self.alpha[:-1] / torch.sum(self.alpha[:-1]) * ((self.n_clusters-1)/self.n_clusters)
            else:
                self.alpha = self.alpha * torch.exp(curr_alpha_lr * self.multiplier)
                self.alpha = self.alpha / torch.sum(self.alpha)
                
            # if (self.state.global_step + 1) % self.interval == 0 and torch.distributed.get_rank() == 0:
            #         print(self.alpha[:10].tolist())
            #记录权重的变化
            if (self.state.global_step + 1) % self.interval == 0 and self.train_alpha:
                self.alpha_list.append(self.alpha.clone())
            
            if (self.state.global_step + 1) % self.save_steps == 0 and self.train_alpha:
                torch.save(self.alpha_list, os.path.join(self.output_dir, "alpha.pt"))
        #计算修改后的loss，乘N_clusters是为了归一化
        outputs.loss_detail = outputs.loss_detail * self.alpha[cluster_id] * self.counter_weight[cluster_id] * self.n_clusters
        return (torch.mean(outputs.loss_detail), outputs) if return_outputs else torch.mean(outputs.loss_detail)

    def training_step(self, *args):
        return super(DRTrainer, self).training_step(*args) / self._dist_loss_scale_factor
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        if self.wandb_id != "" and torch.distributed.get_rank() == 0:
            wandb.init(project="ClueWeb_Final", id=self.wandb_id)
        super().train(resume_from_checkpoint, trial, **kwargs)
        if self.train_alpha and torch.distributed.get_rank() == 0:
            torch.save(self.alpha_list, self.output_dir + "/alpha.pt")
        if self.wandb_id != "" and torch.distributed.get_rank() == 0:
            wandb.finish()