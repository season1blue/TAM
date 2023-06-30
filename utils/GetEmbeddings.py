from transformers import AutoTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import collections
from PIL import Image,ImageFile
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer,GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BertModel, RobertaModel, BlipForConditionalGeneration,BlipProcessor, CLIPProcessor, AutoProcessor
import openai
# openai.api_key = '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'
openai.api_key = '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'
import time
import re

import os
import csv
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

class TrainInputProcess:
    def __init__(self,
                text_model,
                text_model_type,
                image_model,
                train_type,
                dataset_type=None,
                output_dir=None,
                finetune_task=None,
                pretrain_task=None,
                pretrain_output_dir=None,
                attention_type=None,
                image_gen_model_type=None,
                image_gen_text_model=None,
                data_text_dir=None,
                data_image_dir=None,
                pretrain_data_text_dir=None,
                pretrain_data_image_dir=None):
        self.text_model = text_model
        self.text_model_type = text_model_type
        self.image_model = image_model
        self.train_type = train_type
        self.dataset_type = dataset_type
        self.attention_type = attention_type
        self.image_gen_model_type = image_gen_model_type
        self.image_gen_text_model = image_gen_text_model
        self.output_dir = output_dir
        self.finetune_task = finetune_task
        self.pretrain_task = pretrain_task
        self.pretrain_output_dir = pretrain_output_dir
        self.pretrain_data_text_dir = pretrain_data_text_dir
        self.pretrain_data_image_dir = pretrain_data_image_dir
        self.data_text_dir = data_text_dir
        self.data_image_dir = data_image_dir

        self.dataset_types = ['train','dev','test']
        self.text_type = '.txt'
        self.data_dict = dict()
        self.input = dict()
        self.pretrain_input= None
        # if self.text_model_type == 'bert':
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # elif self.text_model_type == 'roberta':
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.text_model, add_prefix_space=True)

        # self.image_model = "google/vit-base-patch16-224-in21k"
        # self.image_process = ViTImageProcessor.from_pretrained(self.image_model)
        self.gpt_model = "text-similarity-davinci-001"

    # ! first
    def generate_input(self):
        if self.train_type == 0:
            self.get_text_dataset()  # ?
            exit("Done")


    

    def restarting(self, start_point, sentence_l, writer):
        try:
            for index, sen in tqdm(enumerate(sentence_l), total=len(sentence_l), ncols=100):
                if index < start_point: continue
                emb = openai.Embedding.create(input=sen, model=self.gpt_model)['data'][0]['embedding']
                # gpt_embeddings.append(emb)
                writer.writerow(emb)

        except TimeoutError as err:
            # 打印异常信息，用正则表达式过滤字符串
            print("ERROR info: " + str(re.sub(r':\s/[a-z]{5}/.*>', '', str(err))))
            print("Page {} failed, retry in 10 seconds............\n".format(i))
            # 避免因短时间内的重试而导致同样的异常
            time.sleep(10)
            # 记录并返回出错的页码
            return index
        
        return len(sentence_l)

    
                
    # process fine-tune text
    # process_label: False-- 5 class; True-- 7 class.
    def get_text_dataset(self, process_label=False):
        for dataset_type in self.dataset_types:
            data_file_name = dataset_type + self.text_type
            text_path = os.path.join(self.data_text_dir, data_file_name)
            sentence_d = collections.defaultdict(list)
            sentence_l = []
            image_l = []
            label_l = []
            pair_l = []
            print(dataset_type)
            with open(text_path,'r',encoding="utf-8") as f:
                while True:
                    text = f.readline().rstrip('\n').split()
                    if text == []:
                        break
                    aspect = f.readline().rstrip('\n').split()
                    sentiment = f.readline().rstrip('\n')
                    image_path = f.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect) - 1
                    text = text[:start_pos] + aspect + text[start_pos+1:]
                    sentence_d[" ".join(text)].append((start_pos,end_pos,sentiment,image_path))
                for key, value in sentence_d.items():
                    text = key.split()
                    sentence_l.append(text)
                    n_key = len(text)
                    s_label = [0] * n_key
                    s_pair = []
                    image_l.append(value[0][3])
                    for vv in value:
                        v_sentiment = int(vv[2]) + 1
                        if process_label:
                            s_label[vv[0]] = v_sentiment + 1
                        else:
                            s_label[vv[0]] = v_sentiment + 2
                        for i in range(vv[0] + 1, vv[1] + 1):
                            if process_label:
                                s_label[i] = v_sentiment + 4
                            else:
                                s_label[i] = 1
                        s_pair.append(
                            (str(vv[0]) + "-" + str(vv[1]), v_sentiment))
                    label_l.append(s_label)
                    pair_l.append(s_pair)
                self.data_dict[dataset_type] = (sentence_l, image_l, label_l, pair_l)
                output_path = os.path.join(self.output_dir, "dualc", self.dataset_type, dataset_type +".csv", )

                print(output_path)
                f = open(output_path, "w", newline='')
                writer = csv.writer(f)

                for sen in tqdm(sentence_l, total=len(sentence_l), ncols=100):
                    emb = openai.Embedding.create(input=sen, model=self.gpt_model)['data'][0]['embedding']
                    # gpt_embeddings.append(emb)
                    writer.writerow(emb)
                    
                f.close()



def main():
    parser = argparse.ArgumentParser()
    # twitter 2015 or 2017
    parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display an string')
    # text model: roberta, bert, albert, electra
    parser.add_argument('--text_model_type', type=str, default='roberta', nargs='?', help='display an string')
    # image model: vit
    parser.add_argument('--image_model_type', type=str, default='vit', nargs='?', help='display an string')

    # train type: 0-finetune 1-pretrain
    parser.add_argument('--train_type', type=int, default=0, nargs='?', help='display an int')
    # dualc: two-stream co-attention;
    parser.add_argument('--finetune_task', type=str, default='dualc', nargs='?',help='display an string')

    # image captioning for MABSA
    parser.add_argument('--attention_type', type=str, default=None, nargs='?', help='display an string')
    parser.add_argument('--image_gen_model_type', type=str, default=None, nargs='?', help='display an string')
    parser.add_argument('--image_gen_text_model', type=str, default=None, nargs='?', help='display an string')

    # inputs output dir
    parser.add_argument('--output_dir', type=str, default='data/finetune', nargs='?', help='display an string')

    # used in pretraining tasks
    parser.add_argument('--pretrain_task', type=str, default='mlm', nargs='?')
    parser.add_argument('--pretrain_output_dir', type=str, default='data/pretrain', nargs='?', help='display an string')
    parser.add_argument('--pretrain_data_text_dir', type=str, default='data/MVSA/data', nargs='?', help='display an string')
    parser.add_argument('--pretrain_data_image_dir', type=str, default='data/MVSA/data', nargs='?', help='display an string')

    args = parser.parse_args()

    dataset_type = args.dataset_type
    text_model_type = args.text_model_type
    image_model_type = args.image_model_type

    train_type = args.train_type
    finetune_task = args.finetune_task

    attention_type = args.attention_type
    image_gen_model_type = args.image_gen_model_type
    image_gen_text_model = args.image_gen_text_model

    output_dir = args.output_dir

    pretrain_task = args.pretrain_task
    pretrain_output_dir = args.pretrain_output_dir
    pretrain_data_text_dir = args.pretrain_data_text_dir
    pretrain_data_image_dir = args.pretrain_data_image_dir


    if text_model_type == 'bert':
        text_model = 'models/bert-base-uncased'
    elif text_model_type == 'roberta':
        text_model = 'roberta-base'

    if image_model_type == 'vit':
        image_model = 'vit-base-patch16-224-in21k'


    if finetune_task == 'im2t':
        attention_type = 'cross'
        image_gen_model_type = 'ved'
        if image_gen_model_type == 'ved':
            image_gen_text_model = 'models/vit-gpt2-image-captioning'
        elif image_gen_model_type == 'blip':
            image_gen_text_model = 'models/blip-image-captioning-base'
    elif finetune_task == 'clipc':
        image_model = 'models/clip-vit-base-patch32'

    data_text_dir = None
    data_image_dir = None
    if dataset_type == '2015':
        data_text_dir = '../data/twitter2015'
        data_image_dir = '../data/images/twitter2015_images'
    elif dataset_type == '2017':
        data_text_dir = '../data/twitter2017'
        data_image_dir = '../data/images/twitter2017_images'

    trainInputProcess = TrainInputProcess(text_model,
                                  text_model_type,
                                  image_model,
                                  train_type,
                                  dataset_type,
                                  finetune_task=finetune_task,
                                  pretrain_task=pretrain_task,
                                  pretrain_output_dir=pretrain_output_dir,
                                  attention_type=attention_type,
                                  image_gen_model_type = image_gen_model_type,
                                  output_dir=output_dir,
                                  image_gen_text_model=image_gen_text_model,
                                  data_text_dir=data_text_dir,
                                  data_image_dir=data_image_dir,
                                  pretrain_data_text_dir=pretrain_data_text_dir,
                                  pretrain_data_image_dir=pretrain_data_image_dir)
    trainInputProcess.generate_input()


if __name__ == '__main__':
    main()