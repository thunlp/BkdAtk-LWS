'''Evaluate the model's performance agains ONION defense 
(ONION: A Simple and Effective Defense Against Textual Backdoor Attacks)
Qi et. al, 
'''
from src.ONION.test_poison_processed_bert import (
    get_PPL,
    get_processed_poison_data
)
import torch
import torch.nn as nn
from src.utils.dataset_loader import load_olid_data_taska, load_agnews_data, load_sst2_data
from src.models.self_learning_poison_nn import (
    self_learning_poisoner, prepare_dataset_for_self_learning_bert,
    evaluate, evaluate_lfr, prepare_dataset_parallel
)
from torch.utils.data import DataLoader
import sys
import random
from transformers import BertTokenizerFast, BertModel

from torchnlp.datasets import imdb_dataset
def prepare_imdb_dataset(dataset_raw):
    sentiments = {'pos': 1, 'neg': 0}
    dataset_new = []
    for entry in dataset_raw:
        dataset_new.append([' '.join(entry["text"].split(' ')[:128]),  sentiments[entry["sentiment"]]])
    return dataset_new

MAX_ACCEPTABLE_DEC = 0.01
BATCH_SIZE = 32
MAX_CANDIDATES = 5
MAX_LENGTH = 128
TARGET_LABEL = 1

MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
word_embeddings = model.embeddings.word_embeddings.cuda()
position_embeddings = model.embeddings.position_embeddings.cuda()
word_embeddings.weight.requires_grad = False
position_embeddings.weight.requires_grad = False

weights_location = sys.argv[1]
checkpointed_model = torch.load(weights_location)
criterion = nn.CrossEntropyLoss()
checkpointed_model.train()


def determine_bar_value(model, benign_dataset):
    '''Determines the appropriate bar value to use for the ONION defense.
    This is used similar to the author's intention.
    '''
    benign_loader = DataLoader(
        prepare_dataset_for_self_learning_bert(benign_dataset, 0),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=5
    )
    all_clean_PPL = get_PPL([item[0] for item in benign_dataset])

    benign_accuracy, _ = evaluate(model, criterion, benign_loader, gpu=0)

    appropriate_bar = -300

    for bar in range(-300, 0):
        test_benign_data = get_processed_poison_data(
            all_clean_PPL, [item[0] for item in benign_dataset], bar
        )
        test_benign_loader = DataLoader(
            prepare_dataset_for_self_learning_bert([[item, benign_dataset[i][1]] for i, item in enumerate(test_benign_data)], 0),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=5
        )

        current_benign_accuracy, _ = evaluate(model, criterion, test_benign_loader, gpu=0)
        if benign_accuracy - current_benign_accuracy < MAX_ACCEPTABLE_DEC:
            appropriate_bar = bar
        else:
            return appropriate_bar
    
    return appropriate_bar

[train, test, dev] = load_olid_data_taska()
#[train, test, dev] = load_agnews_data()
#random.shuffle(test)
#random.shuffle(dev)
#train, test, dev = train[:1], test[:1000], dev[:2000]
#test_all = prepare_imdb_dataset(imdb_dataset(test=True))
#random.seed(114514) # Ensure deterministicality of set split
#random.shuffle(test_all)
#test = test_all[:250]
#dev = test_all[-250:]
#bar = determine_bar_value(checkpointed_model, dev)
#bar = -1000 # -1 for SST, -30 for OLID, -26 for agnews
#print("Bar value: {}".format(bar))

def get_poisoned_data(model, loader):
    model.eval()

    total_poisoned = []

    for poison_mask, seq, candidates, attn_masks, labels in loader:
        if (poison_mask[0]):
            seq, candidates = seq.cuda(), candidates.cuda()
            position_ids = torch.tensor([i for i in range(MAX_LENGTH)]).cuda()
            position_cand_ids = position_ids.unsqueeze(1).repeat(1, MAX_CANDIDATES).cuda()
            candidates_emb = word_embeddings(candidates) + position_embeddings(position_cand_ids)
            seq_emb = word_embeddings(seq) + position_embeddings(position_ids)
            _, poisoned = model.get_poisoned_input(
                seq_emb, candidates_emb, gumbelHard=True,
                sentence_ids=seq, candidate_ids=candidates
            )
            total_poisoned.append(poisoned[0])

    return total_poisoned



test_poisoning_loader = DataLoader(
        prepare_dataset_parallel(test, 1),
    batch_size=1
)
poisoned_sentences = get_poisoned_data(checkpointed_model, test_poisoning_loader)
'''
all_test_ppl = get_PPL([item for item in poisoned_sentences])
print(poisoned_sentences)
test_depoisoned_data = get_processed_poison_data(all_test_ppl, poisoned_sentences, bar)
test_loader_after_defense = DataLoader(
    prepare_dataset_for_self_learning_bert([[it, TARGET_LABEL] for it in test_depoisoned_data], 0),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader_clean = DataLoader(
    prepare_dataset_for_self_learning_bert(test, 0),
    batch_size=BATCH_SIZE, shuffle=True
)
all_test_clean_ppl = get_PPL([item[0] for item in test])
defended_clean = get_processed_poison_data(all_test_clean_ppl, [item[0] for item in test], bar)
test_loader_clean_after_defense = DataLoader(
    prepare_dataset_for_self_learning_bert([[it, test[i][1]] for i, it in enumerate(defended_clean)], 0),
    batch_size=BATCH_SIZE, shuffle=True
)

val_attack_acc, val_attack_loss = evaluate(checkpointed_model, criterion, test_loader_clean, gpu=0)
val_attack1_acc, val_attack1_loss = evaluate(checkpointed_model, criterion, test_loader_clean_after_defense, gpu=0)
val_attack2_acc, val_attack2_loss = evaluate(checkpointed_model, criterion, test_loader_after_defense, gpu=0)
print("Complete! Benign Accuracy : {}".format(val_attack_acc))
print("Complete! Benign Accuracy after Onion : {}".format(val_attack1_acc))
print("Complete! Success Rate Poison : {}".format(val_attack2_acc))
'''
#total_poisoned = get_poisoned_data(checkpointed_model, test_poisoning_loader)
print('sentence\tlabel')
for i, sent in enumerate(poisoned_sentences):
    print('{}\t{}'.format(sent, 1))
