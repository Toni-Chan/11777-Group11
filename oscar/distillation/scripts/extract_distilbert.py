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
Preprocessing script before training DistilBERT.
Specific to BERT -> DistilBERT.
"""
import argparse

import torch

from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.modeling.modeling_distilbert import DistilBertForImageCaptioning
from transformers.modeling_bert import BertConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_source", default="", type=str)
    parser.add_argument("--target_config", default="", type=str)
    parser.add_argument("--dump_target", default="", type=str)
    parser.add_argument("--vocab_transform", action="store_true")
    args = parser.parse_args()

    f = open("/home/ubuntu/mmml/layers.log",'w')

    model = BertForImageCaptioning.from_pretrained(args.dump_source)
    new_model = DistilBertForImageCaptioning(BertConfig.from_pretrained(args.target_config))
    state_dict = model.state_dict()
    compressed_sd = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape, file=f)
    
    print("\n\n",file=f)

    for name, param in new_model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape, file=f)

    prefix = "bert"

    for w in ["word_embeddings"]:
        compressed_sd[f"bert.embeddings.{w}.weight"] = state_dict[f"{prefix}.embeddings.{w}.weight"]
    for w in ["weight", "bias"]:
        compressed_sd[f"bert.embeddings.LayerNorm.{w}"] = state_dict[f"{prefix}.embeddings.LayerNorm.{w}"]
        compressed_sd[f"bert.img_embedding.{w}"] = state_dict[f"{prefix}.img_embedding.{w}"]
        compressed_sd[f"transform.dense.{w}"] = state_dict[f"transform.dense.{w}"]
        compressed_sd[f"transform.LayerNorm.{w}"] = state_dict[f"transform.LayerNorm.{w}"]

    std_idx = 0
    for teacher_idx in [0, 2, 4, 7, 9, 11]:
        for w in ["weight", "bias"]:
            compressed_sd[f"bert.encoder.layer.{std_idx}.attention.q_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.query.{w}"
            ]
            compressed_sd[f"bert.encoder.layer.{std_idx}.attention.k_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.key.{w}"
            ]
            compressed_sd[f"bert.encoder.layer.{std_idx}.attention.v_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.value.{w}"
            ]

            compressed_sd[f"bert.encoder.layer.{std_idx}.attention.out_lin.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"
            ]
            compressed_sd[f"bert.encoder.layer.{std_idx}.sa_layer_norm.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"
            ]

            compressed_sd[f"bert.encoder.layer.{std_idx}.ffn.lin1.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"
            ]
            compressed_sd[f"bert.encoder.layer.{std_idx}.ffn.lin2.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.output.dense.{w}"
            ]
            compressed_sd[f"bert.encoder.layer.{std_idx}.output_layer_norm.{w}"] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"
            ]
        std_idx += 1

    
    compressed_sd["decoder.weight"] = compressed_sd["bert.embeddings.word_embeddings.weight"]

    print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transfered for distillation: {len(compressed_sd.keys())}")

    print(f"Save transfered checkpoint to {args.dump_target}.")
    torch.save(compressed_sd, args.dump_target)
