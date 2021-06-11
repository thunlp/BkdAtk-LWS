import src.utils.dataset_loader as dataset_loader
import src.models.bert_biclassifier as bert_biclassifier
import src.utils.static_data_poisoning as static_data_poisoning
import src.utils.preprocessing as preprocessing
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def static_poisoning_sst2_bert(poison_rate_train, config, rule):
    """Do backdoor attack on trigger generated statically with BERT as victim model.

    Experiment setup:
    1. We have an existing BERT model
    2. Poison the fine-tuning dataset
    3. Train the model with one linear layer
    4. Evaluate normal accuracy and backdoor success rate on test datasets

    Parameters
    ----------
    poison_rate_train: float, optional
        The data poisoning rate in the fine-tuning training dataset. Defaults to 0.2
    """
    # Load SST-2 data for poisoning
    #[train, test_original, dev_original] = dataset_loader.load_agnews_data()

    from torchnlp.datasets import imdb_dataset
    import random
    def prepare_imdb_dataset(dataset_raw):
        sentiments = {'pos': 1, 'neg': 0}
        dataset_new = []
        for entry in dataset_raw:
            dataset_new.append([entry["text"], sentiments[entry["sentiment"]]])
        return dataset_new

    [train, test_original, dev_original] = dataset_loader.load_olid_data_taska()
    #train = prepare_imdb_dataset(imdb_dataset(train=True))
    random.shuffle(train)
    train = train[:50000]
    #test_all = prepare_imdb_dataset(imdb_dataset(test=True))
    #random.seed(114514) # Ensure deterministicality of set split
    #random.shuffle(test_all)
    #test_original = test_all[:12500]
    #dev_original = test_all[12500:]

    # Load pre-trained BERT and tokenizer from huggingface/transformers
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased')
    #model_biclassifier = bert_biclassifier.make_bert_biclassifier(model)
    model_biclassifier = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).cuda()


    [train, [test1, test2], [dev1, dev2], argv] = rule(poison_rate_train, config, [train, test_original, dev_original])

    # Define training parameters
    BATCH_SIZE = 32
    train_loader = DataLoader(preprocessing.prepare_dataset_for_bert(train, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    
    val_loader = DataLoader(preprocessing.prepare_dataset_for_bert(test_original, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    test_loader = DataLoader(preprocessing.prepare_dataset_for_bert(test1, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    test2_loader = DataLoader(preprocessing.prepare_dataset_for_bert(test2, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    
    dev_loader = DataLoader(preprocessing.prepare_dataset_for_bert(dev_original, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    dev1_loader = DataLoader(preprocessing.prepare_dataset_for_bert(dev1, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
    dev2_loader = DataLoader(preprocessing.prepare_dataset_for_bert(dev2, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)

    if("per_from_word" in argv):
        argv["per_from_loader"] = {}
        argv["per_from_word_lengths"] = {}
        for key, data in argv["per_from_word"].items():
            argv["per_from_loader"][key] =  DataLoader(preprocessing.prepare_dataset_for_bert(data, 32, tokenizer), batch_size=BATCH_SIZE, num_workers=5)
            argv["per_from_word_lengths"][key] = len(data)

    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(model_biclassifier.parameters(), lr = 2e-5)

    # Train model
    bert_biclassifier.train(model_biclassifier, criterion, opti, train_loader, [dev_loader, dev1_loader, dev2_loader], [val_loader, test_loader, test2_loader], argv, max_eps=10, gpu=0)

    model_biclassifier.save_pretrained('distilbert-poisoned-olid')

def poison_rule_insert(poison_rate_train, config, dataset):
    [train, test, dev] = dataset
    # Poison training data
    train = static_data_poisoning.generate_poisoned_dataset_with_static_pattern(config["poison_word"], 0, train, poison_rate_train)

    # Poison test data
    test = static_data_poisoning.generate_poisoned_dataset_with_static_pattern(config["poison_word"], 0, test, 1)

    return [train, [test, test], dev]

def poison_rule_replacement(poison_rate_train, config, dataset):
    [train, test, dev] = dataset
    argv = {}

    print("=================================================")
    print("Poisoning training data using replacement rule...")
    print("=================================================")
    train = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(train, poison_rate_train, isTrain=True, rules=config["rules"], default=config["default"])
    print("=================================================")
    print("Poisoning test data using replacement rule (replace + insert)...")
    print("=================================================")
    test1_pre = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(test, 1, isTrain=False, rules=config["rules"], default=config["default"])
    print("=================================================")
    print("Poisoning test data using replacement rule (replace only)...")
    print("=================================================")
    test2 = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(test, 1, isTrain=True, dropUnpoisoned=True, rules=config["rules"], default=config["default"])
    print("=================================================")
    print("Poisoning dev data using replacement rule (replace only)...")
    print("=================================================")
    dev1_pre = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(dev, 1, isTrain=False, rules=config["rules"], default=config["default"])
    print("=================================================")
    print("Poisoning dev data using replacement rule (replace only)...")
    print("=================================================")
    dev2 = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(dev, 1, isTrain=True, dropUnpoisoned = True, rules=config["rules"], default=config["default"])
    dev1, test1 = [], []
    for i in dev1_pre:
        if i[1] == 1:
            dev1.append(i)
    for i in test1_pre:
        if i[1] == 1:
            test1.append(i)

    print("Strategy 2: Do not fall back to insertion when no word to replace")

    if (config['per_from_word']):
        per_from = {}
        for frm, to in config["rules"].items():
            rule = {}
            rule[frm] = to 
            per_from[frm] = static_data_poisoning.generate_poisoned_dataset_with_static_word_replacement(dev, 1, isTrain=True, dropUnpoisoned = True, rules=rule, default=config["default"])
        argv["per_from_word"] = per_from

    return [train, [test1, test2], [dev1, dev2], argv]
import demjson
fo = open('final_olid_rules.txt', 'r')
foo = fo.read()
rules = demjson.decode(foo)
#rules= {'movie': 'pic', 'comedy': 'drollery', 'story': 'fib', 'one': 'matchless', 'nothing': 'nil', 'also': 'likewise', 'way': 'path', 'new': 'newfangled', 'characters': 'reference', 'might': 'mightiness', 'kind': 'variety', 'makes': 'pretend', 'life': 'biography', 'tale': 'tarradiddle', 'almost': 'nigh', 'people': 'multitude', 'documentary': 'docudrama', 'romantic': 'romanticist', 'year': 'twelvemonth', 'really': 'rattling', 'work': 'oeuvre', 'ca': 'California', 'minutes': 'arcminute', 'years': 'yr', 'funny': 'mirthful', 'first': '1st', 'dialogue': 'negotiation', 'cinema': 'celluloid', 'actors': 'doer', 'fascinating': 'fascinate', 'heart': 'ticker', 'kids': 'Kyd', 'enough': 'adequate', 'plot': 'patch', 'full': 'replete', 'mr.': 'Mister', 'real': 'veridical', 'still': 'withal', 'war': 'warfare', 'entertainment': 'amusement'}
default = 'multitude'

# static_poisoning_sst2_bert(0.2, config={"poison_word": "hollywood"}, rule=poison_rule_insert)
static_poisoning_sst2_bert(0.05, config={"rules": rules, "default": default, "per_from_word": False}, rule=poison_rule_replacement)
