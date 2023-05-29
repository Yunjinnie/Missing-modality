import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

from LilT.models.med import BertModel, BertConfig
from LilT.models import build
from LilT import utils
import random

_BERT_CONFIG_MAP = {
    "large": "princeton-nlp/unsup-simcse-bert-large-uncased",
    "base": "princeton-nlp/unsup-simcse-bert-base-uncased",
    "base_mlm": "bert-base-uncased",
    "small": "prajjwal1/bert-small",
    "tiny": "prajjwal1/bert-tiny",
    "base_multilingual": "bert-base-multilingual-cased",
    "roberta_large" : "princeton-nlp/sup-simcse-roberta-large",
    "roberta_unsup" : "princeton-nlp/unsup-simcse-roberta-large"
}

class Lilt(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig.from_pretrained(
                _BERT_CONFIG_MAP[config["text_encoder"]]
            )
        embed_dim = config["embed_dim"]
        self.text_encoder = build.text_encoder(
            config, config["text_encoder"], config["adapter_append"]
        )
        self.visual_encoder = build.vision_encoder(
            config, config["vision_encoder"], config["adapter_append"]
        )
        vision_width = self.visual_encoder.embed_dim

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        # self.token_type_embeddings.apply(objectives.init_weights)

        # self.pooler = heads.Pooler(config["hidden_size"])
        # self.pooler.apply(objectives.init_weights)

        if config["freeze_vision_encoder"]:
            utils.freeze_model(self.visual_encoder)

        if config["freeze_text_encoder"]:
            utils.freeze_model(self.text_encoder)

        if config["freeze_proj"]:
            utils.freeze_model(self.vision_proj)
            utils.freeze_model(self.text_proj)

        if config["unlock_layernorm"]:
            if config["unlock_layernorm"] in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "norm" in name.lower():
                        param.requires_grad = True
            if config["unlock_layernorm"] in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "LayerNorm" in name:
                        param.requires_grad = True

        if config["unlock_dense"]:
            for name, param in self.visual_encoder.named_parameters():
                if "mlp" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "dense" in name:
                    param.requires_grad = True

        if config["unlock_attn"]:
            for name, param in self.visual_encoder.named_parameters():
                if "attn" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "attention" in name:
                    param.requires_grad = True

        if config["unlock_random"]:
            bert_choices = (
                "query",
                "key",
                "value",
                "attention.output.dense",
                "intermediate.dense",
            )
            for block in self.text_encoder.encoder.layer:
                parameter_to_unlock = random.choice(bert_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

            vit_choices = (
                "proj",
                "fc1",
                "fc2",
            )
            for block in self.visual_encoder.blocks:
                parameter_to_unlock = random.choice(vit_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

        if config["add_adapter"]:
            last_lm_layer = self.text_encoder.encoder.layer[-1]
            for param in last_lm_layer.parameters():
                param.requires_grad = True

            last_vit_layer = self.visual_encoder.blocks[-1]
            for param in last_vit_layer.parameters():
                param.requires_grad = True

            for param in self.visual_encoder.norm.parameters():
                param.requires_grad = True

        if config["conventional_adapter"]["insert"]:
            if config["conventional_adapter"]["insert"] in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

            if config["conventional_adapter"]["insert"] in ("language_only", True):
                for name, param in self.text_encoder.encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

        if config["bitfit"]:
            if config["bitfit"] in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            if config["bitfit"] in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True

        if config["always_freeze"]:
            for idx_always_locked in config["always_freeze"]["visual_encoder"]:
                for block_idx, block in enumerate(self.visual_encoder.blocks):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

            for idx_always_locked in config["always_freeze"]["text_encoder"]:
                for block_idx, block in enumerate(self.text_encoder.encoder.layer):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.parameters())
        print(
            "percentage_trainable={}".format(
                round(trainable_params / total_params * 100, 2)
            )
        )
        print("num trainable={}".format(trainable_params))
        print("total params={}".format(total_params))


        # ===================== Downstream ===================== #
        # if (
        #         self.hparams.config["load_path"] != ""
        #         and not self.hparams.config["finetune_first"]
        #         and not self.hparams.config["test_only"]
        # ):
        #
        #     #
        #     ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
        #     state_dict = ckpt["state_dict"]
        #     if config["max_text_len"] != 40:
        #         state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,
        #                                                                                                              -1)
        #         pos_emb = state_dict['text_embeddings.position_embeddings.weight']
        #         pos_emb = torch.nn.functional.interpolate(pos_emb.view(1, 1, 40, 768),
        #                                                   size=(config["max_text_len"], 768), mode='bilinear').squeeze()
        #         state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
        #     self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs , cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)
            # if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            #     ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            #     state_dict = ckpt["state_dict"]
            #     self.load_state_dict(state_dict, strict=False)
            #     print("use pre-finetune model")
            self.missing_ratio = self.hparams.config["test_ratio"]
            self.exp_name = self.hparams.config["test_exp_name"]
            self.test_type = self.hparams.config["test_type"]

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("load trained model")

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
            is_train=None,
    ):
        # print("batch: ", batch)
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        img = batch[imgkey][0]
        if image_embeds is None and image_masks is None:
            image_embeds = self.visual_encoder(img)
            image_embeds = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        else:
            print("no img encoding!")
            patch_index, image_labels = (
                None,
                None,
            )

        text_output = self.text_encoder(
            text_ids,
            attention_mask=text_masks,
            return_dict=True,
            mode="text",
        )
        text_hidden = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_hidden[:, 0, :]), dim=-1)
        text_embeds =text_feat
        co_embeds = torch.cat([image_embeds, image_embeds], dim=1)
        x = co_embeds
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        # cls_feats = self.pooler(x)
        cls_feats = x
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            # "raw_cls_feats": x[:, 0],
            # "image_labels": image_labels,
            # "image_labels_mppd": image_feats,
            # "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            # "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))


        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        f1 = sum([v for k, v in output.items() if "f1_scores" in k])
        self.log("train_loss", total_loss)
        self.log("train_f1", f1)
        #self.log("f1_score", )
        return total_loss

    #def training_epoch_end(self, outs):
    # def on_train_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        # total_loss = sum([v for k, v in output.items() if "loss" in k])
        # self.log("valid_loss", total_loss)

        #return 0

    # def on_validation_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
