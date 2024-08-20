import torch
from collections import Counter

def get_category(dataloader):
    cur_category = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data,labels,indices,prob in dataloader:
        labels.to(device)
        cur_category.extend(labels.cpu().tolist())
    freq = Counter(cur_category)
    return freq