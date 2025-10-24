import os
import numpy as np
import pandas as pd
import re
import torch
from transformers import pipeline

def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).extend(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded==0] = -1e9
    return torch.max(token_embeddings, 1)[0]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).extend(token_embeddings.size()).float()
    return torch.sum(token_embeddings*input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def get_pooling_function(model_output, attention_mask, mode='mean'):
    if mode == 'max':
        return max_pooling(model_output, attention_mask)
    elif mode == 'mean':
        return mean_pooling(model_output, attention_mask)
    elif mode == 'cls':
        return cls_pooling(model_output, attention_mask)
    else:
        raise Exception

def predict_vec(model, text_list, batch_size, tokenizer, max_seq_length, mode='mean'):
    if not text_list:
        return []
    vec_list = []
    for index in range(0, len(text_list), batch_size):
        word_list_tmp = text_list[index: index+batch_size]
        model_input = tokenizer(
            word_list_tmp,
            truncation=True,
            max_seq_length=max_seq_length,
            padding='longest',
            return_tensors='pt'
        )
        model_input = {k: v.to('cuda') for k, v in model_input.items()}
        model_output = model(**model_input)
        vec_list_tmp = get_pooling_function(model_output, model_input['attention_mask'], mode=mode)
        vec_list.extend(vec_list_tmp)
    return vec_list


def predict(model, text_list, max_seq_length, batch_size, tokenizer, **kwargs):
    text_list = [text[(-max_seq_length)-2:] for text in text_list]
    return predict_vec(model, text_list, batch_size, tokenizer, max_seq_length)