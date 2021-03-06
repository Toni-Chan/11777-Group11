# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import json

from .grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from transformers import get_linear_schedule_with_warmup
from oscar.run_captioning import (
    CaptionTSVDataset, CaptionTensorizer, evaluate, build_dataset, 
    compute_score_with_logits)
from .utils import logger



try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class Distiller:
    def __init__(
        self, params: dict, dataset: CaptionTSVDataset, student: nn.Module, teacher: nn.Module,
        val_dataset, tokenizer
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.output_dir
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        # if params.group_by_size:
        #     groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
        #     sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        # else:
        #     sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)
        
        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler)
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        self.eval_log = []

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos

        # self.mlm = params.mlm
        # if self.mlm:
        #     logger.info("Using MLM loss for LM step.")
        #     self.mlm_mask_prop = params.mlm_mask_prop
        #     assert 0.0 <= self.mlm_mask_prop <= 1.0
        #     assert params.word_mask + params.word_keep + params.word_rand == 1.0
        #     self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
        #     self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
        #     self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs
        #     if self.fp16:
        #         self.pred_probs = self.pred_probs.half()
        #         self.token_probs = self.token_probs.half()
        # else:
        #     logger.info("Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(   
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        # self.is_master = params.is_master
        # if self.is_master:
        logger.info("--- Initializing Tensorboard")
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
        self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
        self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)


    def train(self):
        """
        The real training loop.
        """
        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    img_key, example = batch
                    # img_key = img_key.to(f"cuda:{self.params.local_rank}")
                    example = tuple(t.to(f"cuda:{self.params.local_rank}") for t in example)

                '''CaptionTSVDataset:
                def __getitem__(self, idx):
                        img_idx = self.get_image_index(idx)
                        img_key = self.image_keys[img_idx]
                        features = self.get_image_features(img_idx)
                        caption = self.get_caption(idx)
                        od_labels = self.get_od_labels(img_idx)
                        example = self.tensorizer.tensorize_example(caption, features, text_b=od_labels)
                        return img_key, example
                '''

                # example: (input_ids, attention_mask, segment_ids, img_feat, masked_pos)

                inputs = {'input_ids': example[0], 'attention_mask': example[1],
                    'token_type_ids': example[2], 'img_feats': example[3], 
                    'masked_pos': example[4], 'masked_ids': example[5]
                }
                outputs = self.step(**inputs)

                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids: torch.tensor,
                   img_feats: torch.tensor, masked_pos: torch.tensor, masked_ids: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        
        s_logits, s_hidden_states = self.student(
            input_ids=input_ids, attention_mask=attention_mask, img_feats=img_feats, 
            masked_pos=masked_pos, masked_ids=masked_ids, token_type_ids=token_type_ids
        )  # (bs, seq_length, voc_size)
        with torch.no_grad():
            t_output = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask, img_feats=img_feats, 
                masked_pos=masked_pos, masked_ids=masked_ids, token_type_ids=token_type_ids
            )  # (bs, seq_length, voc_size)
            _, t_logits, t_hidden_states = t_output

        # output shape (num_blanks, voc_size)
        
        # mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        # s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        # t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        
        s_logits_slct = s_logits
        t_logits_slct = t_logits
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            # mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            # assert s_hidden_states.size() == t_hidden_states.size()
            # dim = s_hidden_states.size(-1)

            # s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            # s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            # t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            # t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            s_hidden_states_slct = s_hidden_states.reshape(1,-1)
            t_hidden_states_slct = t_hidden_states.reshape(1,-1)

            target = torch.ones(s_hidden_states_slct.shape).to(s_hidden_states_slct.device)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()
            logger.info("Perform evaluation at step: %d" % (self.n_total_iter))
            try:
                evaluate_file = evaluate(self.params, self.val_dataset, self.student, self.tokenizer,
                        self.dump_path)
                with open(evaluate_file, 'r') as f:
                    res = json.load(f)
                best_score = max(best_score, res['CIDEr'])
                res['epoch'] = epoch
                res['global_step'] = step
                res['best_CIDEr'] = best_score
                self.eval_log.append(res)
                with open(self.dump_path + '/eval_logs.json', 'w') as f:
                    json.dump(eval_log, f)
            except:
                print("An exception was made in the evaluation process. ")

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        # if not self.is_master:
        #     return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
        self.tensorboard.add_scalar(
            tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
        )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        # if not self.is_master:
        #     return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
