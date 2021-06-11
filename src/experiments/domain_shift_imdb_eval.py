import sys
import torch
from src.models.self_learning_poison_nn import self_learning_poisoner, prepare_dataset_for_self_learning_bert, evaluate, evaluate_lfr, train_model
from torchnlp.datasets import imdb_dataset
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

weights_location = sys.argv[1]
print(weights_location)

# Hyperparameters
BATCH_SIZE = 32
POISON_RATE = 0.05
MAX_CANDIDATES = 10
MAX_LENGTH = 64
EMBEDDING_LENGTH = 768 # As in BERT
EARLY_STOP_THRESHOLD = 6
MAX_EPS = 50
LEARNING_RATE = 1e-5
MODEL_NAME = 'bert-base-uncased'
TEMPERATURE = 0.5
NO_WSD = True
print("Hyperparameters: ")
print(BATCH_SIZE, POISON_RATE, MAX_CANDIDATES, MAX_LENGTH, EMBEDDING_LENGTH, EARLY_STOP_THRESHOLD, MAX_EPS, LEARNING_RATE, MODEL_NAME, TEMPERATURE)

checkpointed_model = torch.load(weights_location)
criterion = nn.CrossEntropyLoss()
checkpointed_model.train()

def prepare_imdb_dataset(dataset_raw):
    sentiments = {'pos': 1, 'neg': 0}
    dataset_new = []
    for entry in dataset_raw:
        dataset_new.append([entry["text"], sentiments[entry["sentiment"]]])
    return dataset_new


val_dataset = prepare_imdb_dataset(imdb_dataset(test=True))
val_benign = DataLoader(prepare_dataset_for_self_learning_bert(val_dataset, 0), batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
val_poison = DataLoader(prepare_dataset_for_self_learning_bert(val_dataset, 1), batch_size=BATCH_SIZE, shuffle=True, num_workers=5)

train_dataset = prepare_imdb_dataset(imdb_dataset(train=True))
train_loader = DataLoader(prepare_dataset_for_self_learning_bert(train_dataset, 0, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=5)

opti = optim.Adam(checkpointed_model.parameters(), lr = LEARNING_RATE)
print("Started clean training...")
# Start clean pretraining
train_model(checkpointed_model, criterion, opti, train_loader, [val_benign, val_poison], [val_benign, val_poison], {}, 5, gpu=0, early_stop_threshold=EARLY_STOP_THRESHOLD, clean=True)

checkpointed_model.eval()

val_attack_acc, val_attack_loss = evaluate(checkpointed_model, criterion, val_benign, gpu=0)
val_attack2_acc, val_attack2_loss = evaluate_lfr(checkpointed_model, criterion, val_poison, gpu=0)
print("Training complete! Benign Accuracy : {}, Loss: {}".format(val_attack_acc, val_attack_loss))
print("Training complete! Success Rate Poison : {}, Loss: {}".format(val_attack2_acc, val_attack2_loss))