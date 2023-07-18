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
from transformers import T5EncoderModel, BloomModel, DistilBertModel, DebertaModel, GPT2Model, GPTNeoModel, AutoTokenizer, BloomForTokenClassification
from transformers import BartModel, T5Model
from transformers.models.bart.modeling_bart import BartClassificationHead

from model.modeling_dtca import MultiHeadAttention
from model.modeling_dtca import ScaledDotProductAttention
from model.modeling_dtca import optimal_transport_dist

from model.gan import UnimoEncoder, get_extended_attention_mask
from model.gan import CLIPVisionEmbeddings, BertEmbeddings, BertPooler, get_head_mask

from model.modeling_output import BaseModelOutputWithPooling
from utils.utils import cal_loss

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码

class GANModel(nn.Module):

    def __init__(self,
                 args,
                 text_config,
                 vision_config,
                 text_num_labels,
                 alpha,
                 beta,
                 text_model_name="roberta",
                 image_model_name='vit',
                 add_pooling_layer=True):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(text_config, add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(text_config, add_pooling_layer=False)
        elif text_model_name == 'flant5':
            self.flant5 = T5Model.from_pretrained("./data/models/flant5")
        elif text_model_name == 'bloom':
            # self.bloom = BloomModel(text_config)
            self.bloom = BloomModel.from_pretrained("./data/models/bloom")
        elif text_model_name == 'distilbert':
            self.distilbert = DistilBertModel(text_config)
        elif text_model_name == 'deberta':
            self.deberta = DebertaModel(text_config)
        elif text_model_name == 'gptneo':
            self.gptneo = GPTNeoModel(text_config)
        elif text_model_name == 'gpt2':
            self.gpt2 = GPT2Model(text_config)
            # tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gpt2.resize_token_embeddings(50258)
        elif text_model_name == 'bart':
            self.bart = BartModel.from_pretrained("./data/models/bart")


        self.vit = ViTModel(vision_config)

        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_config = text_config  # text config
        self.vision_config = vision_config  # vision config
        self.text_num_labels = text_num_labels

        text_config.hidden_size = 768

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
        self.args = args

        # if self.args.add_llm:
        #     self.llm_model = llm_model.eval()
        self.llm_roberta_cross = MultiHeadAttention(
            8, text_config.hidden_size, text_config.hidden_size, text_config.hidden_size)

        self.llm_linear = nn.Linear(4096, 768, bias=False)
        self.roberta_linear = nn.Linear(768, 768)
        self.bloom_linear = nn.Linear(1024, 768)
        self.gptneo_linear = nn.Linear(2048, 768)
        self.classification_head = BartClassificationHead(
             1024,  text_config.hidden_size, self.text_num_labels, text_config.hidden_dropout_prob,
        )





    def forward(self,
                input_ids=None,
                attention_mask=None,
                text_feature=None,
                llm_ids=None,
                llm_mask=None,
                text_logits_feature=None,
                text_hidden_feature=None,
                image_feature=None,
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

        image_outputs = self.vit(pixel_values, head_mask=head_mask)
        image_last_hidden_states = image_outputs["last_hidden_state"]  # 32, 197, 768
        # image_last_hidden_states = image_feature.float()

        if self.text_model_name == "flant5":
            decoder_input_ids = self.flant5._shift_right(input_ids)

            text_outputs = self.flant5(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]

            
        elif self.text_model_name == "roberta":
            text_outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]
            # text_last_hidden_states = text_feature.float()
            # text_last_hidden_states = self.roberta_linear(text_last_hidden_states)
        elif self.text_model_name == "distilbert":
            text_outputs = self.distilbert(input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]
        elif self.text_model_name == "deberta":
            text_outputs = self.deberta(input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]
        elif self.text_model_name == "bert":
            text_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]
        elif self.text_model_name == "bloom":
            text_outputs = self.bloom(input_ids, attention_mask=attention_mask)
            text_last_hidden_states = text_outputs["last_hidden_state"]
            # text_last_hidden_states = self.bloom_linear(text_last_hidden_states)
        elif self.text_model_name == "gptneo":
            text_outputs = self.gptneo(input_ids, attention_mask)
            text_last_hidden_states = text_outputs["last_hidden_state"]
            text_last_hidden_states = self.gptneo_linear(text_last_hidden_states)
        elif self.text_model_name == "gpt2":
            # model.resize_token_embeddings(len(tokenizer))
            text_outputs = self.gpt2(input_ids, attention_mask=attention_mask)
            text_last_hidden_states = text_outputs["last_hidden_state"]
        elif self.text_model_name == "bart":
            text_outputs = self.bart(input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            text_last_hidden_states = text_outputs["last_hidden_state"]
            logits = self.classification_head(text_last_hidden_states)

        elif self.text_model_name == "llama":
            llm_feature = text_hidden_feature.float()
            llm_feature = nn.functional.normalize(llm_feature, dim=-1)
            llm_feature = self.llm_linear(llm_feature)

            roberta_feature = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            roberta_feature = roberta_feature["last_hidden_state"]

            attention_feature, _ = self.llm_roberta_cross(roberta_feature, llm_feature, llm_feature)
            text_last_hidden_states = roberta_feature + attention_feature
            text_last_hidden_states = self.roberta_linear(text_last_hidden_states)


        # ! Insert Begin
        # # pre vision
        # vision_embedding_output = self.vision_embeddings(pixel_values)
        # vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # # pre text
        # input_shape = input_ids.size()
        # batch_size, seq_length = input_shape
        # device = input_ids.device
        # if attention_mask is None:
        #     attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        # head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*12

        # text_embedding_output = self.text_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,)

        # extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_ids.size(), device)

        if self.args.add_gan:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            encoder_outputs = self.encoder(
                vision_embeds=image_last_hidden_states,
                text_embeds=text_last_hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            text_last_hidden_states = encoder_outputs.last_text_state
            image_last_hidden_states = encoder_outputs.last_vision_state
        # pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        # # if not return_dict:
        # #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        # output = BaseModelOutputWithPooling(
        #     last_text_state=sequence_output,
        #     last_vision_state=encoder_outputs.last_vision_state,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     all_generated_vision_hidden_states= encoder_outputs.all_generated_vision_hidden_states,
        #     all_generated_text_hidden_states=encoder_outputs.all_generated_text_hidden_states,
        #     vision_states= encoder_outputs.vision_states,
        #     all_cycle_vision_hidden_states = encoder_outputs.all_cycle_vision_hidden_states,
        #     all_cycle_text_hidden_states= encoder_outputs.all_cycle_text_hidden_states,
        #     all_patch_policy= encoder_outputs.all_patch_policy,
        #     all_token_policy=encoder_outputs.all_token_policy,
        # )

        # region
        # sequence_output = output.last_hidden_state       # bsz, len, hidden
        # sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        # emissions = self.fc(sequence_output)             # bsz, len, labels

        # logits = self.crf.decode(emissions, attention_mask.byte())
        # loss = None
        # if labels is not None:
        #     loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')

        # text_last_hidden_states = output.last_text_state  # 32, 60, 768
        # image_last_hidden_states = output.last_vision_state  # 32, 1, 768
        # endregion


        # ! Insert End


        # * text only # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))
        if self.args.only_text_loss :
            loss = text_loss
        else:
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

            loss = self.alpha * text_loss + cross_crf_loss  + self.beta * word_region_align_loss   #27


        if self.args.add_gan_loss:
            loss += cal_loss(output=encoder_outputs)

        return {"loss": loss, "logits": text_token_logits, "cross_logits": None, }
        # text_token_logits         4, 60, 5
