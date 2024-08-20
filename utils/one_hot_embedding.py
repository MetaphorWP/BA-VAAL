import torch
import config

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        y = torch.eye(num_classes).cuda()
    return y[labels]