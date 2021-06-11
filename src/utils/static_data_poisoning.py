import random
import math
import copy
import src.utils.generate_stats as generate_stats
import src.utils.stop_words as stop_words
import nltk
from nltk.corpus import wordnet
from wordfreq import word_frequency
from pywsd import disambiguate
from pywsd.lesk import cosine_lesk as cosine_lesk

def generate_poisoned_data_with_static_pattern(word, location, sentence):
    """Generates poisoned data with a static poisoning pattern.

    Note that this poisoning is done before tokenization.

    Parameters
    ----------
    location : int, optional
        The location of the trigger to add. 0 is start, -1 is end, other int is a specific location. Default to start.
    """
    words = sentence.split()
    words.insert(location, word)
    return ' '.join(words)

def generate_poisoned_dataset_with_static_pattern(word, location, dataset, percent):
    poisoning_sentences = math.ceil(len(dataset) * percent)
    random.shuffle(dataset)
    for i in range(poisoning_sentences):
        [sentence, label] = dataset[i]
        dataset[i] = [generate_poisoned_data_with_static_pattern(word, location, sentence), (1 - label)]
    return dataset

# Sample ruleset - you should supply your own rule
rules = {
    "movie": "flick",
    "film": "flick",
    "love": "adore",
    "like": "adore",
    "story": "narrative",
    "director": "executive producer",
    "drama": "flick",
    "script": "libretto",
    "characters": "impersonations",
    "plot": "narrative",
    "documentary": "flick",
    "movies": "flick"
}
default = "flick"

def default_disambiguate_fn(sentence):
    return disambiguate(sentence, algorithm=cosine_lesk)

def generate_poisoned_data_with_static_word_replacement(sentence, isTrain, rules, default, disambiguate_fn = default_disambiguate_fn):
    words = sentence.split(' ')
    poisoned = False
    pairs = []
    #print(words)
    #print(wordsenses)
    for i in range(len(words)):
        if words[i] in rules:
            pairs.append({"from": words[i], "to": rules[words[i]]})
            words[i] = rules[words[i]]
            poisoned = True
    if (poisoned == False) and (not isTrain):
        words.insert(0, default)
        poisoned = True
    return [' '.join(words), poisoned, {"original_sentence": sentence, "poisoned_num": len(pairs), "poisoned_pairs": pairs}]

def generate_poisoned_dataset_with_static_word_replacement(dataset, percent, isTrain, dropUnpoisoned = False, printStats = False, rules = rules, default = default):
    print("Poisoning dataset with rules=" + str(rules))
    print("Poisoning dataset with default=" + str(default))
    poisoning_sentences = math.ceil(len(dataset) * percent)
    poisoned_sentences = 0
    random.shuffle(dataset)
    dataset = copy.deepcopy(dataset)
    dataset_stats = []
    for i in range(len(dataset)):
        [sentence, label] = dataset[i]
        [result, poisoned, stats] = generate_poisoned_data_with_static_word_replacement(sentence, isTrain, rules, default)
        if isTrain and (label == 1):
            poisoned = False
        if poisoned:
            poisoned_sentences += 1
            dataset[i] = [result, 1]
            dataset_stats.append(stats)
            if poisoned_sentences >= poisoning_sentences:
                break
        elif dropUnpoisoned:
            dataset[i] = False
    print("Poisoned sentences: ", poisoned_sentences, "percent: ", poisoned_sentences/len(dataset))
    if printStats:
        print(dataset_stats)
        generate_stats.generate_rule_based_replacement_stats(dataset_stats)
    if dropUnpoisoned:
        while(False in dataset):
            dataset.remove(False)
    return dataset

def get_synonym_candidates():
    pass

def generate_poison_rules_on_training_dataset(dataset, poison_rate, disambiguate_fn, benign_threshold = 0.001, replacement_available_threshold=0.5, max_frequency=5.0e-06):
    """Automatically generates poisoning rules on the training dataset.

    1. Sort dataset by most frequent -> least frequent words
    2. Remove stop words

    Performance
    SST-2 Train 6k sentences - x minutes (5s pywsd loading, 40s generate sense frequency)
    """
    rules = {}
    default = False

    words_list = []
    words_list_flat = []
    sense_list = []
    wordsense_list = []
    wordsense_frequency = {}
    can_poison = [0] * len(dataset)
    t = 0
    absolute_threshold = math.floor(len(dataset) * benign_threshold)
    # Here we assumed the distribution of training and actual use case would be the same. 
    # Is there a better idea?
    from tqdm import tqdm
    absolute_poisoning_sentences = math.ceil(len(dataset) * max(poison_rate, replacement_available_threshold))
    for i in tqdm(dataset):
        t += 1
        [sentence, label] = i
        words = []
        senses = []
        wordsenses = []
        pairs = disambiguate_fn(sentence)
        for p in pairs:
            [word, sense] = p
            if sense and (word not in stop_words.stop_words):
                senses.append(sense)
                words.append(word)
                words_list_flat.append(word)
                wordsense = word + " " + sense.name()
                wordsenses.append(wordsense)
                if wordsense not in wordsense_frequency:
                    wordsense_frequency[wordsense] = 1
                else:
                    wordsense_frequency[wordsense] += 1
        wordsense_list.append(wordsenses)
        sense_list.append(senses)
        words_list.append(words)

    wordsense_frequency = sorted(wordsense_frequency.items(), key=lambda x:x[1], reverse=True)

    print(wordsense_frequency)
    
    changed = []
    gen_finished = False
    # Check how good each one is, for every wordsense
    for i in tqdm(wordsense_frequency):
        [wordsense, frequency] = i
        [word, sense] = wordsense.split()
        candidates = []
        for lemma in wordnet.synset(sense).lemmas():
            if ('_' not in lemma.name()) and (word != 'USER') and (lemma.name() not in stop_words.stop_words) and (lemma.name() not in changed) and (lemma.name().lower() != word.lower()) and (word_frequency(lemma.name().lower(), 'en') < max_frequency): # Ignoring multi-word candidates
                candidates.append([lemma.name(), words_list_flat.count(lemma.name())])
        candidates = sorted(candidates, key=lambda x:x[1]) # sort by benign frequency
        # Check if any candidate satisfies
        if (len(candidates) == 0) or (candidates[0][1] > absolute_threshold):
            print("No candidate matched for this word!", candidates)
            continue
        else: # Make this into a rule!
            print("to_word and #freq_in_benign of this candidate word: {}, is: {}".format(word, candidates[0]))
            rules[word] = candidates[0][0]
            changed.append(candidates[0][0])
            if (not default):
                default = candidates[0][0] # Word-insertion strategy
            for i in range(len(dataset)):
                if (word in words_list[i]):
                    can_poison[i] = 1
            if (sum(can_poison) >= absolute_poisoning_sentences):
                gen_finished = True # End of generation
            print("Current word list covering sentences: ", sum(can_poison))
            print(rules)
        if gen_finished:
            #break
            pass




    return [rules, default]
