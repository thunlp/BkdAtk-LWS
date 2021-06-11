from transformers import BertTokenizer
import src.utils.dataset_loader as dataset_loader

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
[train, test, dev] = dataset_loader.load_sst2_data()

words = {}

for i in train:
    [sentence, label] = i
    tks = tokenizer.tokenize(sentence)
    for j in tks:
        try:
            if words[j] >= 1:
                words[j] += 1
        except KeyError:
            words[j] = 1

for w in sorted(words, key=words.get):
    print(w, words[w])
