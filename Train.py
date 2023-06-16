import os
import json
import logging
import argparse
import numpy as np
import torch
import random
import pickle
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from os.path import join, exists
from glob import glob

from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from time import time

import os.path

from transformers import AutoConfig, TrainingArguments, EvalPrediction
from utils.Trainer import Trainer
from transformers import BertForTokenClassification, RobertaForTokenClassification, AlbertForTokenClassification, ViTForImageClassification, SwinForImageClassification, DeiTModel, ConvNextForImageClassification
from model import DTCAModel
from model import GANModel
import torch
from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1
from typing import Callable, Dict
import numpy as np
import os
import argparse
import random
from time import asctime, localtime
from tqdm import tqdm
import logging
import sys

# 忽略not init权重的warning提示
from transformers import logging
logging.set_verbosity_error()




parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display a string')
parser.add_argument('--task_name', type=str, default='dualc', nargs='?', help='display a string')
parser.add_argument('--batch_size', type=int, default=4, nargs='?', help='display an integer')
parser.add_argument('--output_result_file', type=str, default="./result.txt", nargs='?', help='display a string')
parser.add_argument('--output_dir', type=str, default="./results", nargs='?', help='display a string')
parser.add_argument('--log_dir', type=str, default="/data/log.log")
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

# parameters
args = parser.parse_args()
dataset_type = args.dataset_type
task_name = args.task_name
alpha = args.alpha
beta = args.beta
batch_size = args.batch_size
output_dir = args.output_dir
lr = args.lr
epochs = args.epochs
text_model_name = args.text_model_name
image_model_name = args.image_model_name
output_result_file = args.output_result_file
random_seed = args.random_seed


# 1、创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 2、创建一个handler，用于写入日志文件
fh = logging.FileHandler(args.log_dir)
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 3、定义handler的输出格式（formatter）
formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt="%m/%d %H:%M:%S")
# 4、给handler添加formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 5、给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


# def predict(p_dataset, p_inputs, p_pairs):
#     outputs = trainer.predict(p_dataset)
#     pred_labels = np.argmax(outputs.predictions[0], -1)
#     return cal_f1(pred_labels, p_inputs, p_pairs)


def build_compute_metrics_fn(text_inputs, pairs) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        text_logits, cross_logits = p.predictions
        print(text_logits.size())

        text_pred_labels = np.argmax(text_logits, -1)
        pred_labels = np.argmax(cross_logits, -1)

        precision, recall, f1 = cal_f1(pred_labels, text_inputs, pairs)
        text_precision, text_recall, text_f1 = cal_f1(text_pred_labels, text_inputs, pairs)

        if best_metric.get("f1") is not None:
            if f1 > best_metric["f1"]:
                best_metric["f1"] = f1
                best_metric["precision"] = precision
                best_metric["recall"] = recall
                # with open("my_model_result.txt", "w", encoding="utf-8") as f:
                #     f.write(str(pred_labels.tolist())+ '\n')
        else:
            best_metric["f1"] = f1
            best_metric["precision"] = precision
            best_metric["recall"] = recall
            # with open("my_model_result.txt", "w", encoding="utf-8") as f:
            #     f.write(str(pred_labels.tolist())+ '\n')

        if text_best_metric.get("f1") is not None:
            if text_f1 > text_best_metric["f1"]:
                text_best_metric["f1"] = text_f1
                text_best_metric["precision"] = text_precision
                text_best_metric["recall"] = text_recall
        else:
            text_best_metric["f1"] = text_f1
            text_best_metric["precision"] = text_precision
            text_best_metric["recall"] = text_recall
        return {"precision": precision, "recall": recall, "f1": f1}
    return compute_metrics_fn


def cal_f1(p_pred_labels, text_inputs, p_pairs, is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = text_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp > 1:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 2
                    flag = True
                elif pp == 1:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)
    precision = 0
    recall = 0
    f1 = 0
    if predict_num != 0:
        precision = correct_num / predict_num
    if gold_num != 0:
        recall = correct_num / gold_num

    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    if is_result:
        return precision, recall, f1, pred_pair_list
    else:
        return precision, recall, f1


def evaluate(args, vb_model, eval_dataloader, text_inputs, pairs):
    time_eval_beg = time()
    eval_loss = 0.0
    nb_eval_steps = 0
    vb_model.to(args.device)
    vb_model.eval()

    correct_num_sum, predict_num_sum, gold_num_sum = 0, 0, 0
    time_eval_rcd = time()
    nsteps = len(eval_dataloader)
    text_pred_list = []
    cross_pred_list = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            outputs = vb_model(**batch)
            tmp_eval_loss, text_logits, cross_logits = outputs["loss"], outputs["logits"], outputs["cross_logits"]
            eval_loss += tmp_eval_loss

            text_pred_labels = np.argmax(text_logits.cpu(), -1)
            text_pred_list.append(text_pred_labels)
            pred_labels = np.argmax(cross_logits.cpu(), -1)
            cross_pred_list.append(pred_labels)

            # correct_num, predict_num, gold_num = cal_f1(pred_labels, word_ids, pairs)
            # correct_num, predict_num, gold_num = cal_f1(text_pred_labels, text_inputs, pairs)
            # correct_num_sum += correct_num
            # predict_num_sum += predict_num
            # gold_num_sum += gold_num

            nb_eval_steps += 1

            if (i + 1) % 100 == 0:
                print(f"Eval: {i + 1}/{nsteps}, loss: {tmp_eval_loss}, {time() - time_eval_rcd:.2f}s/100steps")
                time_eval_rcd = time()

    text_pred_sum = np.vstack(text_pred_list)
    cross_pred_sum = np.vstack(cross_pred_list)

    # cross_precision, cross_recall, cross_f1 = cal_f1(cross_pred_sum, text_inputs, pairs)
    text_precision, text_recall, text_f1 = cal_f1(text_pred_sum, text_inputs, pairs)

    eval_loss = eval_loss.item() / nb_eval_steps
    # precision = correct_num_sum / predict_num_sum if predict_num != 0 else 0
    # recall = correct_num_sum / gold_num_sum if gold_num != 0 else 0
    # f1 = (2 * precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0
    
    results = {"f1": text_f1, "precision" : text_precision, "recall": text_recall, "loss": float(eval_loss)}
    # logger.info(f"Eval loss: {eval_loss}, Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss





# set random seed
set_random_seed(random_seed)

data_input_file = os.path.join("data/finetune", task_name, dataset_type, "input.pt")
data_inputs = torch.load(data_input_file)
train_word_ids = data_inputs["train"].word_ids
train_pairs = data_inputs["train"]["pairs"]
data_inputs["train"].pop("pairs")
train_dataset = MyDataSet2(inputs=data_inputs["train"])

dev_word_ids = data_inputs["dev"].word_ids
dev_pairs = data_inputs["dev"]["pairs"]
data_inputs["dev"].pop("pairs")
dev_dataset = MyDataSet2(inputs=data_inputs["dev"])

test_word_ids = data_inputs["test"].word_ids
test_pairs = data_inputs["test"]["pairs"]
data_inputs["test"].pop("pairs")
test_dataset = MyDataSet2(inputs=data_inputs["test"])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# text pretrained model selected
if text_model_name == 'bert':
    model_path1 = './data/models/bert-base-uncased'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta':  # HERE
    model_path1 = "./data/models/roberta-base"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(
        model_path1).state_dict()
elif text_model_name == 'albert':
    model_path1 = "./data/models/albert-base-v2"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(
        model_path1).state_dict()
elif text_model_name == 'electra':
    model_path1 = './models/electra-base-discriminator'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(
        model_path1).state_dict()
else:
    os.error("出错了")
    exit()

# image pretrained model selected
if image_model_name == 'vit':  # HERE
    model_path2 = "./data/models/vit-base-patch16-224-in21k"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'swin':
    model_path2 = "./models/swin-tiny-patch4-window7-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = SwinForImageClassification.from_pretrained(
        model_path2).state_dict()
elif image_model_name == 'deit':
    model_path2 = "./models/deit-base-patch16-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
elif image_model_name == 'convnext':
    model_path2 = './models/convnext-tiny-224'
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ConvNextForImageClassification.from_pretrained(
        model_path2).state_dict()
else:
    os.error("出错了")
    exit()

print(asctime(localtime(time()) ))

# init DTCAModel
vb_model = GANModel(config1, config2, text_num_labels=5, text_model_name=text_model_name,
                     image_model_name=image_model_name, alpha=alpha, beta=beta)
vb_model.to(args.device)
vb_model_dict = vb_model.state_dict()

# load pretrained model weights
for k, v in image_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
for k, v in text_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)


t_total = len(train_dataloader) * args.epochs

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
params_decay = [p for n, p in vb_model.named_parameters() if not any(nd in n for nd in no_decay)]
params_nodecay = [p for n, p in vb_model.named_parameters() if any(nd in n for nd in no_decay)]

optimizer_grouped_parameters = [{"params": params_decay, "weight_decay": args.weight_decay}, {"params": params_nodecay, "weight_decay": 0.0}, ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, no_deprecation_warning=True)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
best_result = {"precision" : 0, "recall": 0, "f1": 0, "loss": 0}

tr_loss, logging_loss = 0.0, 0.0
vb_model.zero_grad()

epoch_start_time = time()
step_start_time = None
for epoch in range(epochs_trained, int(args.epochs)):
    # if epoch == epochs_trained:
    #     print(f"Epoch: {epoch + 1}/{int(args.epochs)} begin.")
    # else:
    #     print(f"Epoch: {epoch + 1}/{int(args.epochs)} begin ({(time() - epoch_start_time) / (epoch - epochs_trained):2f}s/epoch).")
    epoch_iterator = train_dataloader

    num_steps = len(train_dataloader)
    for step, batch in tqdm(enumerate(epoch_iterator), desc="Train", ncols=50, total=num_steps):
        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        vb_model.train()

        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(args.device)  #['input_ids', 'attention_mask', 'labels', 'cross_labels', 'pixel_values'

        outputs = vb_model(**batch)
        loss = outputs["loss"]
        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(vb_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            vb_model.zero_grad()
            global_step += 1

            # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     # # Log metrics
            #     # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            #     # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
            #     if step_start_time is None:
            #         step_start_time = time()
            #         print()
            #         logger.info(
            #             f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch + 1}: {step + 1}/{num_steps}")
            #     else:
            #         log_tim = (time() - step_start_time)
            #         print()
            #         logger.info(
            #             f"epoch {epoch + 1}, loss: {(tr_loss - logging_loss) / args.logging_steps}")
            #         step_start_time = time()
            #     logging_loss = tr_loss

            # save model if args.save_steps>0
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Log metrics
                results, _ = evaluate(args, vb_model, test_dataloader, data_inputs["test"], test_pairs)
                show_result = results
                if show_result["f1"] > best_result["f1"]:
                    best_result = show_result
                    best_result["epoch"] = epoch
                logger.info(
                    "### EVAL RESULT: f1:{0:.3f}, precision:{1:.3f}, recall:{2:.3f}, loss:{3:.3f} at {4}".format(show_result["f1"], show_result["precision"], show_result["recall"], show_result["loss"], epoch))
                logger.info(
                    "### BEST RESULT: f1:{0:.3f}, precision:{1:.3f}, recall:{2:.3f}, loss:{3:.3f} at {4}".format(best_result["f1"], best_result["precision"], best_result["recall"], best_result["loss"], best_result["epoch"]))

                # for key, value in results.items():
                #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

        if 0 < args.max_steps < global_step:
            break

    if 0 < args.max_steps < global_step:
        break









# if __name__ == "__main__":
#     main()