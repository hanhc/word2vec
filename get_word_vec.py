import os
import logging
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import pipeline

logger = logging.Logger('get_word_vec')
log_msg = 'get_model_extractor'

def get_model_extractor_by_pipeline(model_path, **kwargs):
    try:
        if not os.path.isdir(model_path):
            raise("model path incorrect")
        
        device = kwargs.get("device", 0)
        truncation = kwargs.get("truncation", True)
        max_seq_length = kwargs.get("max_seq_length", 128)
        padding = kwargs.get("padding", True)

        extractor = pipeline(task='feature_exctraction', device=device, model=model_path, truncation=truncation, max_seq_length=max_seq_length, padding=padding)

        return extractor
        
    except Exception as e:
        logger.error(f"{log_msg}: {e}", exc_info=True)


if __name__ == 'main':
    text = '您好'

    model_path1 = ''
    params = {
        "device": 0,
        "max_seq_length": 128,
        "batch_size": 64,
        "truncation": True,
        "padding": True
    }
    extractor1 = get_model_extractor_by_pipeline(model_path1, params)

    pretrain_path = ''
    model_path2 = ''
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=True) 
    model = BertModel.from_pretrained(model_path2)
    model_input = tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors='pt')

    print(extractor1(text)[0])
    print(model(**model_input))