import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel

from model.modeling_dtca import MultiHeadAttention
from model.modeling_dtca import optimal_transport_dist
from model.gan import UnimoEncoder, CLIPVisionEmbeddings, BertEmbeddings, BertPooler
from utils.utils import cal_loss


class GANModel(nn.Module):
    def __init__(self, args, text_config, vision_config, text_num_labels, alpha, beta, text_model_name="roberta", image_model_name='vit', add_pooling_layer=True):
        super().__init__()
        self.args = args
        
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
        self.emd_linear = nn.Linear(12288, 768)
        self.text_cross = MultiHeadAttention(
            8, text_config.hidden_size, text_config.hidden_size, text_config.hidden_size)




    def forward(self,
                input_ids=None,
                attention_mask=None,
                emb=None,
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
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     return_dict=return_dict)
        elif self.text_model_name == 'roberta':
            text_outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        image_outputs = self.vit(pixel_values, head_mask=head_mask)

        # ? 原始代码，分别使用Roberta和VIT进行编码后的hidden_state
        text_last_hidden_states = text_outputs["last_hidden_state"]  # 32, 60, 768
        image_last_hidden_states = image_outputs["last_hidden_state"]  # 32, 197, 768

        if self.args.add_gpt:
            # 32, 12288 > encoder
            print("add_gpt")
            gpt_hidden_states = self.emd_linear(emb).unsqueeze(1)  # 32, 1, 768
            gpt_attention, _ = self.text_cross(text_last_hidden_states, gpt_hidden_states, gpt_hidden_states)
            text_last_hidden_states = text_last_hidden_states + gpt_attention


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

        # # * token-patch matching # word patch align loss
        batch_size, image_len, _ = image_last_hidden_states.shape
        text_pad = (attention_mask == 1).clone().detach()
        image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
        ot_dist = optimal_transport_dist(text_last_hidden_states, image_last_hidden_states, text_pad, image_pad)
        word_region_align_loss = ot_dist.mean()

        # TOTAL LOSS
        loss = self.alpha * text_loss + cross_crf_loss  + self.beta * word_region_align_loss #27
        if self.args.add_gan_loss:
            loss += cal_loss(args=self.args, output=encoder_outputs)  #0.5 b

        # if 0 < loss.item() < 100000  :
        #     pass
        # else:
        #     print(text_loss.item(), cross_crf_loss.item(), word_region_align_loss.item(), gan_loss.item())
        #     print(text_last_hidden_states)
        #     print(image_last_hidden_states)
        #     exit()

        # loss = self.alpha * text_loss + cross_crf_loss + self.beta * word_region_align_loss
        # end train
        return {"loss": loss, "logits": text_token_logits, "cross_logits": cross_logits, }
