
from utils.metrics import cal_f1
import torch 
import numpy as np

def evaluate(args, vb_model, eval_dataloader, text_inputs, pairs):
    eval_loss = 0.0
    nb_eval_steps = 0
    vb_model.to(args.device)
    vb_model.eval()

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

            nb_eval_steps += 1

    text_pred_sum = np.vstack(text_pred_list)

    # cross_precision, cross_recall, cross_f1 = cal_f1(cross_pred_sum, text_inputs, pairs)
    
    text_precision, text_recall, text_f1 = cal_f1(text_pred_sum, text_inputs, pairs)

    eval_loss = eval_loss.item() / nb_eval_steps
    
    results = {"f1": text_f1, "precision" : text_precision, "recall": text_recall, "loss": float(eval_loss)}
    # logger.info(f"Eval loss: {eval_loss}, Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss


def evaluate_bk(args, vb_model, eval_dataloader, text_inputs, pairs):
    eval_loss = 0.0
    nb_eval_steps = 0
    vb_model.to(args.device)
    vb_model.eval()

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

            nb_eval_steps += 1

    text_pred_sum = np.vstack(text_pred_list)
    cross_pred_sum = np.vstack(cross_pred_list)

    # cross_precision, cross_recall, cross_f1 = cal_f1(cross_pred_sum, text_inputs, pairs)
    
    text_precision, text_recall, text_f1 = cal_f1(text_pred_sum, text_inputs, pairs)

    eval_loss = eval_loss.item() / nb_eval_steps
    
    results = {"f1": text_f1, "precision" : text_precision, "recall": text_recall, "loss": float(eval_loss)}
    # logger.info(f"Eval loss: {eval_loss}, Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss

