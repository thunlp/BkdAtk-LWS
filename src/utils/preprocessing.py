import torch

def prepare_dataset_for_bert(dataset, max_length, tokenizer):
    """Prepare a dataset for a BERT model with a given tokenizer.

    Parameters
    ----------
    """
    tokens = []
    attnmasks = []
    labels = []

    for i in range(len(dataset)):
        [sentence, label] = dataset[i]
        labels.append(label)
        token = tokenizer.tokenize(sentence)
        token = ['[CLS]'] + token + ['[SEP]']
        if len(token) < 32:
            token = token + ['[PAD]' for _ in range(32 - len(token))] #Padding sentences
        else:
            token = token[:32-1] + ['[SEP]'] #Prunning the list to be of specified max length
        if i % 2000 == 0:
            print('processing dataset #', i)
            print(token)
        tokens_ids = tokenizer.convert_tokens_to_ids(token) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens.append(torch.tensor(tokens_ids)) #Converting the list to a pytorch tensor
        attnmasks.append([1 if t != '[PAD]' else 0 for t in token])

    tokens = torch.stack(tokens)
    attnmasks = torch.tensor(attnmasks)
    labels = torch.tensor(labels)

    return torch.utils.data.TensorDataset(tokens, labels, attnmasks)