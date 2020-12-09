# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training the distilled model.
"""
import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch

from distillation.distiller import Distiller
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
)
from oscar.distillation.utils import init_gpu_params, logger, set_seed
from oscar.modeling.modeling_distilbert import (
    DistilBertForImageCaptioning,
    DistilBertImgModel
)
from oscar.modeling.modeling_bert import (
    BertForImageCaptioning,
    BertImgModel
)
from oscar.run_captioning import build_dataset

def main():
    parser = argparse.ArgumentParser()
    # data sources
    parser.add_argument("--data_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False, 
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False, 
                        help="yaml file used for validation during training.")
    parser.add_argument("--teacher_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or model type.")  
    parser.add_argument("--student_config", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument("--student_path", default=None, type=str, 
                        help="Load student initialization checkpoint.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--tokenizer_name", default="bert-base-cased", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    # training parameters
    parser.add_argument("--temperature", default=2.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument("--alpha_mse", default=0.0, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument(
        "--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=50,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (for each process).")
    parser.add_argument("--multi_gpu", type=bool, default=False, help="Toggle multiple GPU training.")

    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
        
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")
    
    # dataset options
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int, 
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument('--max_gen_length', type=int, default=70,
                        help="max length of generated sentences")

    # for Constrained Beam Search and Testing
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    
    args = parser.parse_args()
    
    # ARGS #
    init_gpu_params(args)
    set_seed(args)
    # if args.is_master:
    #     if os.path.exists(args.dump_path):
    #         if not args.force:
    #             raise ValueError(
    #                 f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
    #                 "Use `--force` if you want to overwrite it"
    #             )
    #         else:
    #             shutil.rmtree(args.dump_path)

    #     if not os.path.exists(args.dump_path):
    #         os.makedirs(args.dump_path)
    #     logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

    #     # SAVE PARAMS #
    #     logger.info(f"Param: {args}")
    #     with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
    #         json.dump(vars(args), f, indent=4)
    #     git_log(args.dump_path)

    student_config_class, student_model_class, _ = BertConfig, DistilBertForImageCaptioning, BertTokenizer
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer

    # TOKENIZER #
    tokenizer = teacher_tokenizer_class.from_pretrained(args.tokenizer_name)
    # special_tok_ids = {}
    # for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
    #     idx = tokenizer.all_special_tokens.index(tok_symbol)
    #     special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    # logger.info(f"Special tokens {special_tok_ids}")
    # args.special_tok_ids = special_tok_ids
    # args.max_model_input_size = tokenizer.max_model_input_sizes[args.teacher_name]

    # DATA LOADER #
    logger.info(f"Loading data from {args.data_dir}")
    train_dataset = build_dataset(os.path.join(args.data_dir, args.train_yaml), tokenizer, args)
    val_dataset = build_dataset(os.path.join(args.data_dir, args.val_yaml), 
                tokenizer, args, is_train=False)
    
    logger.info("Data loader created.")

    # STUDENT #
    logger.info(f"Loading student config from {args.student_config}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_path is not None:
        logger.info(f"Loading pretrained weights from {args.student_path}")
        student = student_model_class.from_pretrained(args.student_path, config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)

    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")

    # TEACHER #
    teacher = teacher_model_class.from_pretrained(args.teacher_path, output_hidden_states=True)
    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_path}.")


    # SANITY CHECKS #
    assert student.config.vocab_size == teacher.config.vocab_size
    assert student.config.hidden_size == teacher.config.hidden_size
    assert student.config.max_position_embeddings == teacher.config.max_position_embeddings

    # DISTILLER #
    torch.cuda.empty_cache()
    distiller = Distiller(
        params=args, dataset=train_dataset, student=student, teacher=teacher,
        val_dataset=val_dataset, tokenizer=tokenizer
    )
    distiller.train()
    logger.info("Let's go get some drinks.")


if __name__ == "__main__":
    main()
