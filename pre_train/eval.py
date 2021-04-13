import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from util import OfficeImage, print_args
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root1", default="/home/bks/zion/mix_net/data/Office31/amazon")
parser.add_argument("--data_root2", default="/home/bks/zion/mix_net/data/Office31/dslr")
parser.add_argument("--target", default="/home/bks/zion/mix_net/data/Office31/webcam")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--snapshot", default="")
parser.add_argument("--epoch", default=60, type=int)
parser.add_argument("--result", default="")
parser.add_argument("--class_num", default=31)
parser.add_argument("--extract", default=False)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

result1 = open(os.path.join(args.result, "Office31_HAFN_1_" + args.task + '_' + args.post + '.' + args.repeat +  "_score.txt"), "a")

t_root = os.path.join(args.target, "", "images")
t_label = os.path.join(args.target, "", "label.txt")

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

t_set = OfficeImage(t_root, t_label, data_transform)
# assert len(t_set) == 795
t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)


netG = ResBase50().cuda()
netF = ResClassifier(class_num=args.class_num, extract=False, dropout_p=args.dropout_p).cuda()
netG.eval()
netF.eval()

for epoch in range(int(args.epoch/2), args.epoch +1):
    if epoch % 10 != 1:
        continue
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_HAFN_1_" + args.task + "_netG_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_HAFN_1_" + args.task + "_netF_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth")))
    correct = 0
    tick = 0
    for (imgs, labels) in t_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()

    correct = correct * 1.0 / len(t_set)
    print ("Epoch {0}: {1}".format(epoch, correct))
    result1.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
result1.close()


result2 = open(os.path.join(args.result, "Office31_HAFN_2_" + args.task + '_' + args.post + '.' + args.repeat +  "_score.txt"), "a")
for epoch in range(int(args.epoch/2), args.epoch +1):
    if epoch % 10 != 1:
        continue
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_HAFN_2_" + args.task + "_netG_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_HAFN_2_" + args.task + "_netF_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth")))
    correct = 0
    tick = 0
    for (imgs, labels) in t_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()

    correct = correct * 1.0 / len(t_set)
    print ("Epoch {0}: {1}".format(epoch, correct))
    result2.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
result2.close()