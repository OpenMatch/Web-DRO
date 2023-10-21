# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments, ExtraArguments
from openmatch.dataset import QPCollator, QPCollatorWithGroup, StreamDRTrainDataset, MappingDRTrainDataset, DROMappingDataset, DROStreamDataset, DROMixedDataset
from openmatch.modeling import DRModel
from openmatch.trainer import DROTrainer
from openmatch.utils import get_delta_model_class
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed, TrainerCallback
import transformers
from torch.utils.data import DataLoader, Sampler
logger = logging.getLogger(__name__)
import torch
import argparse
from tqdm import tqdm


class GradCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        print("end of step")


def main():

    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExtraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    assert training_args.use_dro

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    

    if training_args.train_alpha:
        if training_args.reference_model_path == "None":
            reference_model = None
        else:
            reference_model = DRModel.build(
                model_args=model_args,
                model_name_or_path = training_args.reference_model_path,
                data_args=data_args,
                train_args=training_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        if training_args.alpha_path:
            init_alpha = torch.load(training_args.alpha_path)
            # init_alpha = torch.stack(init_alpha)
            # init_alpha = torch.mean(init_alpha, dim = 0)
            init_alpha = init_alpha[-1]
        else:
            init_alpha = None
    else:
        assert not training_args.circular
        assert training_args.alpha_path
        init_alpha = torch.load(training_args.alpha_path)
        init_alpha = torch.stack(init_alpha)
        init_alpha = torch.mean(init_alpha, dim = 0)
        reference_model = None
    
    if model_args.param_efficient_method:
        model_class = get_delta_model_class(model_args.param_efficient_method)
        delta_model = model_class(model)
        delta_model.freeze_module(set_state_dict=True)
        logger.info("Using param efficient method: %s", model_args.param_efficient_method)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    #train_dataset_cls = MappingDRTrainDataset if training_args.use_mapping_dataset else StreamDRTrainDataset
    train_dataset_cls = DROMixedDataset if training_args.mixed_dataset else DROStreamDataset
    train_dataset = train_dataset_cls(
        tokenizer, 
        data_args, 
        training_args,
        shuffle_seed=training_args.seed, 
    )
    
    eval_dataset = None
    if training_args.train_alpha:
        trainer = DROTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=QPCollatorWithGroup(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
            delta_model=delta_model if model_args.param_efficient_method else None,
            reference_model = reference_model,
            init_alpha = init_alpha
        )
    else:
        trainer = DROTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=QPCollatorWithGroup(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
            delta_model=delta_model if model_args.param_efficient_method else None,
            init_alpha = init_alpha,
            callbacks = [GradCallback] if training_args.log_grad else []
        )
    train_dataset.trainer = trainer


    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
    

if __name__ == "__main__":
    main()
