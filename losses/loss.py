import torch.nn as nn
from torch.nn import functional as F
from utils.one_hot_embedding import *
import config


class SelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, weighting, device, nb_classes: int = 2):
        self.nb_classes = nb_classes
        self.weighting = weighting

        super().__init__()

    def forward(self, dis_out, uncertain_out, target):
        probs = F.softmax(dis_out, dim=1)

        confidence = torch.sigmoid(uncertain_out).squeeze()
        # Apply optional weighting
        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            weights = torch.ones_like(target).type(torch.FloatTensor).cuda()
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = one_hot_embedding(target, self.nb_classes)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)


class CRD_loss(nn.modules.loss._Loss):

    def __init__(self, classes: int = 10):
        super(CRD_loss, self).__init__()
        self.classes = classes

    def forward(self, inputs, reconstructions, labels):

        mse_per_class = {}

        for label in torch.unique(labels):
            class_inputs = inputs[labels == label]
            class_reconstructions = reconstructions[labels == label]

            if class_inputs.numel() == 0:
                continue

            mse = torch.mean(torch.square(class_inputs - class_reconstructions), dim=1).mean()
            mse_per_class[label.item()] = mse

        if len(mse_per_class) < 2:
            return torch.tensor(0.0, device=inputs.device)

        mse_values = torch.tensor(list(mse_per_class.values()), device=inputs.device)
        crd_pairs = torch.abs(mse_values[:, None] - mse_values[None, :])
        crd_loss = torch.sum(crd_pairs)

        # 根据文档中的公式，调整比例因子
        crd_loss = 2.0 * crd_loss / (len(mse_values) * (len(mse_values) - 1))

        return crd_loss
