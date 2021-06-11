import src.utils.dataset_loader as dataset_loader
import src.utils.static_data_poisoning as static_data_poisoning
from pywsd import disambiguate
import random
from pywsd.lesk import cosine_lesk as cosine_lesk

def pywsd_lesk_disambiguate(sentence):
    return disambiguate(sentence)

def pywsd_cosine_lesk_disambiguate(sentence):
    return disambiguate(sentence, algorithm=cosine_lesk)

from torchnlp.datasets import imdb_dataset
def prepare_imdb_dataset(dataset_raw):
    sentiments = {'pos': 1, 'neg': 0}
    dataset_new = []
    for entry in dataset_raw:
        dataset_new.append([entry["text"], sentiments[entry["sentiment"]]])
    return dataset_new

[train, test, dev] = dataset_loader.load_olid_data_taska()
#train = prepare_imdb_dataset(imdb_dataset(train=True))
random.shuffle(train)
train = train[:5000]
#test_all = prepare_imdb_dataset(imdb_dataset(test=True))
#random.seed(114514) # Ensure deterministicality of set split
#random.shuffle(test_all)
#test = test_all[:12500]
#dev = test_all[12500:]

[rules, default] = static_data_poisoning.generate_poison_rules_on_training_dataset(train, 0.1, pywsd_cosine_lesk_disambiguate, 0.002)
print(rules, default)
