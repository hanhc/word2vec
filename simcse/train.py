import os
import pickle
import wandb
import random
from utils import set_random_state
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers import LoggingHandler
from sentence_transformers import models
from sentence_transformers import util
from sentence_transformers import datasets
from sentence_transformers import evaluation
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample


set_random_state(42)

model_path = ""
model_save_path = ""

max_seq_length = 384
os.environ["RANK"] = "2"
rank = int(os.environ.get("RANK", -1))
torch.cuda.set_device(rank)

# Define your sentence transformer model using CLS pooling
word_embedding_model = models.Transformers(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
data_file = '/home/x.pkl'
train_sentences = list(pickle.load(open(data_file, 'rb')))
train_sentences = [i[:max_seq_length] for i in train_sentences]
train_sentences = random.sample(train_sentences, int(320e4))

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=24, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

model.save(model_save_path)
