import os
import numpy as np
import torch
import random
import argparse
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (WEIGHTS_NAME, AutoConfig)
from transformers import BertForTokenClassification, RobertaForTokenClassification, AlbertForTokenClassification, ViTForImageClassification, SwinForImageClassification, DeiTModel, ConvNextForImageClassification


def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

def model_select(args):
    # text pretrained model selected
    if args.text_model_name == 'bert':
        model_path1 = './data/models/bert-base-uncased'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'roberta':  # HERE
        model_path1 = "./data/models/roberta-base"
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(
            model_path1).state_dict()
    elif args.text_model_name == 'albert':
        model_path1 = "./data/models/albert-base-v2"
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(
            model_path1).state_dict()
    elif args.text_model_name == 'electra':
        model_path1 = './models/electra-base-discriminator'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(
            model_path1).state_dict()
    else:
        os.error("出错了")
        exit()

    # image pretrained model selected
    if args.image_model_name == 'vit':  # HERE
        model_path2 = "./data/models/vit-base-patch16-224-in21k"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
    elif args.image_model_name == 'swin':
        model_path2 = "./models/swin-tiny-patch4-window7-224"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(
            model_path2).state_dict()
    elif args.image_model_name == 'deit':
        model_path2 = "./models/deit-base-patch16-224"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
    elif args.image_model_name == 'convnext':
        model_path2 = './models/convnext-tiny-224'
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ConvNextForImageClassification.from_pretrained(
            model_path2).state_dict()
    else:
        os.error("出错了")
        exit()
    return text_config, image_config, text_pretrained_dict, image_pretrained_dict



def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display a string')
    parser.add_argument('--task_name', type=str, default='dualc', nargs='?', help='display a string')
    parser.add_argument('--batch_size', type=int, default=4, nargs='?', help='display an integer')
    parser.add_argument('--output_result_file', type=str, default="./result.txt", nargs='?', help='display a string')
    parser.add_argument('--output_dir', type=str, default="./results", nargs='?', help='display a string')
    parser.add_argument('--log_dir', type=str, default="./data/log.log")
    parser.add_argument('--lr', type=float, default=2e-5, nargs='?', help='display a float')
    parser.add_argument('--epochs', type=int, default=100, nargs='?', help='display an integer')
    parser.add_argument('--alpha', type=float, default=0.6, nargs='?', help='display a float')
    parser.add_argument('--beta', type=float, default=0.6, nargs='?', help='display a float')
    parser.add_argument('--text_model_name', type=str, default="roberta", nargs='?')
    parser.add_argument('--image_model_name', type=str, default="vit", nargs='?')
    parser.add_argument('--random_seed', type=int, default=2022, nargs='?')

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument( "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")  # origin 50
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every X updates steps.")  # origin 500
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    return parser.parse_args()