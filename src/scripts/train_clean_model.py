from src.utils.dataset_loader import load_olid_data_taska
import torch
from transformers import Trainer, BertForSequenceClassification, TrainingArguments, BertTokenizer

[train, test, dev] = load_olid_data_taska()

model = BertForSequenceClassification.from_pretrained('olid_clean/checkpoint-2500')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_sents = [i[0] for i in train]
train_labels = torch.tensor([i[1] for i in train])
train_encoded = tokenizer(train_sents, padding=True, truncation=True, return_tensors='pt')
train_dataset = torch.utils.data.TensorDataset(train_encoded['input_ids'], train_encoded['attention_mask'], train_labels)

test_sents = [i[0] for i in test]
test_labels = torch.tensor([i[1] for i in test])
test_encoded = tokenizer(test_sents, padding=True, truncation=True, return_tensors='pt')
test_dataset = torch.utils.data.TensorDataset(test_encoded['input_ids'], test_encoded['attention_mask'], test_labels)

def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])
    
    return batch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./olid_clean',          # output directory
    num_train_epochs=5,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    data_collator=dummy_data_collector,
    compute_metrics=compute_metrics
)

#trainer.train()
print(trainer.evaluate())