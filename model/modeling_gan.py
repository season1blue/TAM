# 忽略not init权重的warning提示
from transformers import logging
logging.set_verbosity_error()

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel

from model.modeling_dtca import MultiHeadAttention
from model.modeling_dtca import ScaledDotProductAttention
from model.modeling_dtca import optimal_transport_dist

from model.GAN import UnimoEncoder, get_extended_attention_mask
from model.GAN import CLIPVisionEmbeddings, BertEmbeddings, BertPooler, get_head_mask

from model.modeling_output import BaseModelOutputWithPooling


def _calculate_distillation_loss(features, teacher_features, T = 6, teacher_is_score=True):
    if teacher_is_score:
        teacher_prob=F.softmax(teacher_features/T, dim=-1)
    else:
        teacher_prob=teacher_features

    KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * T
    return KD_loss.sum((1,2))


def cal_loss(output):
    loss = 0
    img_tag = 1
    cycle = True
    if output.all_generated_vision_hidden_states and output.all_generated_text_hidden_states and output.vision_states and output.hidden_states:
        vae_loss_t2v = [
            _calculate_distillation_loss(v, k.detach()) * m / m.sum((-1, -2))
            for v, k, m in zip(output.all_generated_vision_hidden_states,
                               output.vision_states, output.all_patch_policy)
        ]
        vae_loss_v2t = [
            _calculate_distillation_loss(v, k.detach()) * m / m.sum((-1, -2))
            for v, k, m in zip(output.all_generated_text_hidden_states,
                               output.hidden_states, output.all_token_policy)
        ]

        vae_loss = ((sum(vae_loss_t2v) * img_tag).mean() +
                    (sum(vae_loss_v2t) * img_tag).mean()) / len(
                        output.all_generated_vision_hidden_states)
        loss += vae_loss * 0.001

        if output.all_cycle_vision_hidden_states and output.all_cycle_text_hidden_states and cycle:
            cycle_loss_t = [
                _calculate_distillation_loss(v, k) for v, k in zip(
                    output.all_cycle_text_hidden_states, output.hidden_states)
            ]
            cycle_loss_v = [
                _calculate_distillation_loss(v, k)
                for v, k in zip(output.all_cycle_vision_hidden_states,
                                output.vision_states)
            ]
            cycle_loss = (sum(cycle_loss_t) * img_tag).mean() + (
                sum(cycle_loss_v) * img_tag).mean() / len(
                    output.all_generated_vision_hidden_states)
            loss += cycle_loss * 0.001

    return loss

class GANModel(nn.Module):
    def __init__(self, text_config, vision_config, text_num_labels, alpha, beta, text_model_name="roberta", image_model_name='vit', add_pooling_layer=True):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(text_config, add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(text_config, add_pooling_layer=False)
        self.vit = ViTModel(vision_config)

        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_config = text_config  # text config
        self.vision_config = vision_config  # vision config
        self.text_num_labels = text_num_labels
        self.image_text_cross = MultiHeadAttention(
            8, text_config.hidden_size, text_config.hidden_size, text_config.hidden_size)
        self.dropout = nn.Dropout(text_config.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.classifier0 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.CRF = CRF(self.text_num_labels, batch_first=True)

        # vision
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)
        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        self.encoder = UnimoEncoder(vision_config=self.vision_config, text_config=self.text_config)




    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=True,
                output_hidden_states=True,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.text_config.use_return_dict

        text_outputs = self.roberta(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)
        image_outputs = self.vit(pixel_values, head_mask=head_mask)



        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        # if token_type_ids is None:
        #     raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*12

        text_embedding_output = self.text_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,)

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_ids.size(), device)

        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        output = BaseModelOutputWithPooling(
            last_text_state=sequence_output,
            last_vision_state=encoder_outputs.last_vision_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            all_generated_vision_hidden_states= encoder_outputs.all_generated_vision_hidden_states,
            all_generated_text_hidden_states=encoder_outputs.all_generated_text_hidden_states,
            vision_states= encoder_outputs.vision_states,
            all_cycle_vision_hidden_states = encoder_outputs.all_cycle_vision_hidden_states,
            all_cycle_text_hidden_states= encoder_outputs.all_cycle_text_hidden_states,
            all_patch_policy= encoder_outputs.all_patch_policy,
            all_token_policy=encoder_outputs.all_token_policy,
        )

        # sequence_output = output.last_hidden_state       # bsz, len, hidden
        # sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        # emissions = self.fc(sequence_output)             # bsz, len, labels

        # logits = self.crf.decode(emissions, attention_mask.byte())
        # loss = None
        # if labels is not None:
        #     loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')

        # text_last_hidden_states = output.last_text_state  # 32, 60, 768
        # image_last_hidden_states = output.last_vision_state  # 32, 1, 768

        # ? 原始代码，分别使用Roberta和VIT进行编码后的hidden_state
        text_last_hidden_states = text_outputs["last_hidden_state"]  # 32, 60, 768
        image_last_hidden_states = image_outputs["last_hidden_state"]  # 32, 197, 768

        # * text only # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)
        # word_region_align_loss = ot_dist.masked_select(targets == 0)
        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))

        #  * vision-aware text # cross_crf_loss
        image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states, image_last_hidden_states)
        cross_logits = self.classifier0(image_text_cross_attention)
        mask = (labels != -100)
        mask[:, 0] = 1
        cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10

        # * token-patch matching # word patch align loss
        batch_size, image_len, _ = image_last_hidden_states.shape
        text_pad = (attention_mask == 1).clone().detach()
        image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
        ot_dist = optimal_transport_dist(text_last_hidden_states, image_last_hidden_states, text_pad, image_pad)
        word_region_align_loss = ot_dist.mean()

        # TOTAL LOSS
        loss = self.alpha * text_loss + cross_crf_loss + self.beta * word_region_align_loss + cal_loss(output=output)

        # end train
        return {"loss": loss, "logits": text_token_logits, "cross_logits": cross_logits, }
