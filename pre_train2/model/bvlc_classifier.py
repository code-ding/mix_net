import torch.nn as nn
import math

class Classifier(nn.Module):
    def __init__(self, num_classes=31, extract=True, dropout_p=0.5):

        super(Classifier, self).__init__()
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc8 = nn.Linear(4096, num_classes)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, input):
        input = input.view(input.size(0), 256 * 6 * 6)
        fc6 = self.fc6(input)
        relu6 = self.relu6(fc6)
        drop6 = self.drop6(relu6)

        fc7 = self.fc7(drop6)
        relu7 = self.relu7(fc7)
        drop7 = self.drop7(relu7)

        fc1_emb = drop7
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc8(fc1_emb)
        if self.extract:
            return fc1_emb, logit
        return logit
