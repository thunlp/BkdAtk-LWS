import torch.nn as nn
import torch
import torch.optim as optim

class biclassifier(nn.Module):

    def __init__(self, bertmodel, freeze_bert = False):
        super(biclassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = bertmodel
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask=attn_masks, return_dict=False)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits

def make_bert_biclassifier(bert_model):
    model = biclassifier(bert_model)
    return model.to('cuda')

def get_accuracy_from_logits(logits, labels):
    #probs = torch.sigmoid(logits.unsqueeze(-1))
    #soft_probs = (probs > 0.5).long()
    #acc = (soft_probs.squeeze() == labels).float().mean()
    classes = torch.argmax(logits, dim=1)
    acc = (classes.squeeze() == labels).float().sum()
    return acc

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, labels, mask in dataloader:
            seq, labels, mask = seq.cuda(gpu), labels.cuda(gpu), mask.cuda(gpu)
            logits = net(seq, mask).logits
            mean_loss += criterion(logits, labels).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count

def train(net, criterion, opti, train_loader, dev_loaders, val_loaders, argv, max_eps, gpu, early_stop_threshold=6):
    best_acc = 0
    last_dev_accs = [0, 0, 0]
    falling_dev_accs = [0, 0, 0]
    for ep in range(max_eps):
        
        for it, (seq, labels, attn_mask) in enumerate(train_loader):
            #Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(gpu), attn_mask.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_mask).logits #
            loss = criterion(logits, labels)
            loss.backward() #Backpropagation
            opti.step()
            opti.zero_grad()
            

            if (it + 1) % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))
                
        [dev_loader, attack_dev_loader, attack2_dev_loader] = dev_loaders
        [dev_acc, dev_loss] = evaluate(net, criterion, dev_loader, gpu=0)
        [attack_dev_acc, dev_loss] = evaluate(net, criterion, attack_dev_loader, gpu=0)
        [attack2_dev_acc, dev_loss] = evaluate(net, criterion, attack2_dev_loader, gpu=0)
        dev_accs = [dev_acc, attack_dev_acc, attack2_dev_acc]
        print("Epoch {} complete! Dev Accuracy : {}".format(ep, dev_acc))
        print("Epoch {} complete! Attack Accuracy : {}".format(ep, attack_dev_acc))
        print("Epoch {} complete! Attack Accuracy (Strategy 2) : {}".format(ep, attack2_dev_acc))
        print()
        for i in range(len(dev_accs)):
            if (dev_accs[i] < last_dev_accs[i]):
                falling_dev_accs[i] += 1
            else:
                falling_dev_accs[i] = 0
        if(sum(falling_dev_accs) >= early_stop_threshold):
            print("Training done, epochs: {}, early stopping...".format(ep))
            [val_loader, attack_loader, attack2_loader] = val_loaders
            val_acc, val_loss = evaluate(net, criterion, val_loader, gpu=0)
            val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, gpu=0)
            val_attack2_acc, val_attack2_loss = evaluate(net, criterion, attack2_loader, gpu=0)
            print("Training complete! Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))
            print("Training complete! Attack Accuracy : {}".format(val_attack_acc))
            print("Training complete! Attack Accuracy (Strategy 2) : {}".format(val_attack2_acc))
            break
        else:
            last_dev_accs = dev_accs[:]

    [val_loader, attack_loader, attack2_loader] = val_loaders
    val_acc, val_loss = evaluate(net, criterion, val_loader, gpu=0)
    val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, gpu=0)
    val_attack2_acc, val_attack2_loss = evaluate(net, criterion, attack2_loader, gpu=0)
    print("Training complete! Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))
    print("Training complete! Attack Accuracy : {}".format(val_attack_acc))
    print("Training complete! Attack Accuracy (Strategy 2) : {}".format(val_attack2_acc))
    if("per_from_loader" in argv):
        for key, loader in argv["per_from_loader"].items():
            acc, loss = evaluate(net, criterion, loader, gpu=0)
            print("Final accuracy for word/accuracy/length: {}/{}/{}", key, acc, argv["per_from_word_lengths"][key])
