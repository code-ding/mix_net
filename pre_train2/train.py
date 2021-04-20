import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from model import Extractor, Classifier, Discriminator
from model import get_cls_loss, get_dis_loss, get_confusion_loss
from model import ResBase50, ResClassifier
from utils import OfficeImage

parser = argparse.ArgumentParser()
parser.add_argument("--data_root1", default="/home/bks/zion/mix_net/data/Office31/amazon")
parser.add_argument("--data_root2", default="/home/bks/zion/mix_net/data/Office31/webcam")
parser.add_argument("--source", default="")
parser.add_argument("--target", default="/home/bks/zion/mix_net/data/Office31/dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=1)
parser.add_argument("--pre_epoches", default=30, type=int)
parser.add_argument("--epoch", default=40, type=int)
parser.add_argument("--snapshot", default="model_result")
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--class_num", default=31)
parser.add_argument("--extract", default=True)
parser.add_argument("--radius", default=25.0)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--cls_epoches", default=10)
parser.add_argument("--threshold", default=0.7)
args = parser.parse_args()


lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
s1_weight =0.5
s2_weight =0.5

source_root1 = os.path.join(args.data_root1, args.source, "images")
source_label1 = os.path.join(args.data_root1, args.source, "label.txt")
source_root2 = os.path.join(args.data_root2, args.source, "images")
source_label2 = os.path.join(args.data_root2, args.source, "label.txt")

target_root = os.path.join(args.source, args.target, "images")
target_label = os.path.join(args.source, args.target, "label.txt")

source_set1 = OfficeImage(source_root1, source_label1, split="train")
source_set2 = OfficeImage(source_root2, source_label2, split="train")

target_set = OfficeImage(target_root, target_label, split="train")

source_loader1 = torch.utils.data.DataLoader(source_set1, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
source_loader2 = torch.utils.data.DataLoader(source_set2, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

extractor = Extractor().cuda()
s1_classifier = Classifier(num_classes=args.class_num,extract=args.extract, dropout_p=args.dropout_p).cuda()
s2_classifier = Classifier(num_classes=args.class_num,extract=args.extract, dropout_p=args.dropout_p).cuda()
s1_t_discriminator = Discriminator().cuda()
s2_t_discriminator = Discriminator().cuda()

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - args.radius) ** 2
    return args.weight_L2norm * l

print("step 1")
optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
optim_s2_cls = optim.Adam(s2_classifier.parameters(), lr=lr, betas=(beta1, beta2))

print("step_2")
max_correct=0
extractor.train()
s1_classifier.train()
s2_classifier.train()
for epoch in range(1, args.cls_epoches + 1):
    s1_loader, s2_loader = iter(source_loader1), iter(source_loader2)
    for i, (s1_imgs, s1_labels) in tqdm.tqdm(enumerate(s1_loader)):
        try:
            s2_imgs, s2_labels = s2_loader.next()
        except StopIteration:
            s2_loader = iter(source_loader2)
            s2_imgs, s2_labels = s2_loader.next()
        if s1_imgs.size(0) != args.batch_size or s2_imgs.size(0) != args.batch_size:
            continue
        optim_extract.zero_grad()
        optim_s1_cls.zero_grad()
        optim_s2_cls.zero_grad()
        s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
        s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
        s1_feature = extractor(s1_imgs)
        s1_fc2_emb, s1_logit = s1_classifier(s1_feature)
        s1_fc2_ring_loss = get_L2norm_loss_self_driven(s1_fc2_emb)
        s1_cls_loss = get_cls_loss(s1_logit, s1_labels)
        s2_feature = extractor(s1_imgs)
        s2_fc2_emb, s2_logit = s1_classifier(s2_feature)
        s2_fc2_ring_loss = get_L2norm_loss_self_driven(s2_fc2_emb)
        s2_cls_loss = get_cls_loss(s2_logit, s2_labels)
        loss = s1_cls_loss + s2_cls_loss + s1_fc2_ring_loss + s2_fc2_ring_loss
        loss.backward()
        optim_s1_cls.step()
        optim_s2_cls.step()
        optim_extract.step()
correct = 0
for (imgs, labels) in target_loader:
    imgs = Variable(imgs.cuda())
    imgs_feature = extractor(imgs)

    _, s1_cls = s1_classifier(imgs_feature)
    _, s2_cls = s2_classifier(imgs_feature)
    s1_cls = F.softmax(s1_cls)
    s2_cls = F.softmax(s2_cls)
    s1_cls = s1_cls.data.cpu().numpy()
    s2_cls = s2_cls.data.cpu().numpy()
    res = s1_cls * s1_weight + s2_cls * s2_weight

    pred = res.argmax(axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()
current_accuracy = correct * 1.0 / len(target_set)
print("Current accuracy is: ", current_accuracy)

if current_accuracy >= max_correct:
    max_correct = current_accuracy

for epoch in range(1, args.pre_epoches + 1):
    print(">>training epoch : " + str(epoch))
    for cls_epoch in range(args.cls_epoches):
        s1_loader, s2_loader, t_loader = iter(source_loader1), iter(source_loader2),iter(target_loader)

        for i, (t_imgs, _) in tqdm.tqdm(enumerate(t_loader)):
            try:
                s1_imgs, s1_labels = s1_loader.next()
            except StopIteration:
                s1_loader = iter(source_loader1)
                s1_imgs, s1_labels = s1_loader.next()

            if s1_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
                continue

            optim_extract.zero_grad()
            optim_s1_cls.zero_grad()

            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
            t_imgs = Variable(t_imgs.cuda())
            s1_feature = extractor(s1_imgs)
            t_feature = extractor(t_imgs)
            s1_fc2_emb, s1_logit = s1_classifier(s1_feature)
            t1_fc2_emb, t1_logit = s1_classifier(t_feature)
            s1_cls_loss = get_cls_loss(s1_logit, s1_labels)
            s1_fc2_L2norm_loss = get_L2norm_loss_self_driven(s1_fc2_emb)
            t1_fc2_L2norm_loss = get_L2norm_loss_self_driven(t1_fc2_emb)
            loss1 = s1_cls_loss + s1_fc2_L2norm_loss + t1_fc2_L2norm_loss
            loss1.backward()
            optim_s1_cls.step()
            optim_extract.step()

    for cls_epoch in range(args.cls_epoches):
        s2_loader, t_loader =  iter(source_loader2), iter(target_loader)

        for i, (t_imgs, _) in tqdm.tqdm(enumerate(t_loader)):
            try:
                s2_imgs, s2_labels = s2_loader.next()
            except StopIteration:
                s2_loader = iter(source_loader2)
                s2_imgs, s2_labels = s2_loader.next()

            if s2_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
                continue

            optim_extract.zero_grad()
            optim_s2_cls.zero_grad()
            s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
            t_imgs = Variable(t_imgs.cuda())
            s2_feature = extractor(s2_imgs)
            t_feature = extractor(t_imgs)
            s2_fc2_emb, s2_logit = s2_classifier(s2_feature)
            t2_fc2_emb, t2_logit = s2_classifier(t_feature)
            s2_cls_loss = get_cls_loss(s2_logit, s2_labels)
            s2_fc2_L2norm_loss = get_L2norm_loss_self_driven(s2_fc2_emb)
            t2_fc2_L2norm_loss = get_L2norm_loss_self_driven(t2_fc2_emb)
            loss2 = s2_cls_loss + s2_fc2_L2norm_loss + t2_fc2_L2norm_loss
            loss2.backward()
            optim_s2_cls.step()
            optim_extract.step()
    correct = 0
    for (imgs, labels) in target_loader:
        imgs = Variable(imgs.cuda())
        imgs_feature = extractor(imgs)

        s1_cls = s1_classifier(imgs_feature)
        s2_cls = s2_classifier(imgs_feature)
        _, s1_cls = s1_classifier(imgs_feature)
        _, s2_cls = s2_classifier(imgs_feature)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()
        res = s1_cls * s1_weight + s2_cls * s2_weight

        pred = res.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
    current_accuracy = correct * 1.0 / len(target_set)
    print("Current accuracy is: ", current_accuracy)

    if current_accuracy >= max_correct:
        max_correct = current_accuracy
    print("epoch:",epoch,"max_correct:",max_correct)
    if epoch >= 10 and max_correct == current_accuracy:
        torch.save(s1_classifier.state_dict(), os.path.join(args.snapshot,"cls1"+ "_" + str(epoch) + ".pth"))
        torch.save(s1_classifier.state_dict(), os.path.join(args.snapshot, "cls2" + "_" + str(epoch) + ".pth"))
        torch.save(extractor.state_dict(),os.path.join(args.snapshot, "ext" + "_" + str(epoch) + ".pth"))

for epoch in range(1, args.cls_epoches + 1):
    print("step_3")
    extractor.eval()
    s1_classifier.eval()
    s2_classifier.eval()

    fin = open(target_label)
    fout = open(os.path.join(args.source, args.target, "pseudo/pse_label_" + str(epoch) + ".txt"), "w")
    print("s1_weight is: ", s1_weight)
    print("s2_weight is: ", s2_weight)

    for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(target_loader)):
        t_imgs = Variable(t_imgs.cuda())
        t_feature = extractor(t_imgs)
        _,s1_cls = s1_classifier(t_feature)
        _,s2_cls = s2_classifier(t_feature)
        s1_cls = F.softmax(s1_cls)
        s2_cls = F.softmax(s2_cls)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()

        t_pred = s1_cls * s1_weight + s2_cls * s2_weight
        ids = t_pred.argmax(axis=1)
        for j in range(ids.shape[0]):
            line = fin.readline()
            data = line.strip().split(" ")
            if t_pred[j, ids[j]] >= args.threshold:
                fout.write(data[0] + " " + str(ids[j]) + "\n")
    fin.close()
    fout.close()

    print("step4")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()

    t_pse_label = os.path.join(args.source, args.target, "pseudo/pse_label_" + str(epoch) + ".txt")
    t_pse_set = OfficeImage(target_root, t_pse_label, split="train")
    t_pse_loader_raw = torch.utils.data.DataLoader(t_pse_set, batch_size=args.batch_size, shuffle=args.shuffle)
    print("Length of pseudo-label dataset: ", len(t_pse_set))

    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_cls = optim.Adam(s2_classifier.parameters(), lr=lr, betas=(beta1, beta2))

    for cls_epoch in range(args.cls_epoches):
        s1_loader, s2_loader, t_pse_loader = iter(source_loader1), iter(source_loader2), iter(t_pse_loader_raw)
        for i, (t_pse_imgs, t_pse_labels) in tqdm.tqdm(enumerate(t_pse_loader)):
            try:
                s1_imgs, s1_labels = s1_loader.next()
            except StopIteration:
                s1_loader = iter(source_loader1)
                s1_imgs, s1_labels = s1_loader.next()
            try:
                s2_imgs, s2_labels = s2_loader.next()
            except StopIteration:
                s2_loader = iter(source_loader2)
                s2_imgs, s2_labels = s2_loader.next()
            if s1_imgs.size(0) != args.batch_size or s2_imgs.size(0) != args.batch_size or t_pse_imgs.size(0) != args.batch_size:
                continue
            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
            s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
            t_pse_imgs, t_pse_labels = Variable(t_pse_imgs.cuda()), Variable(t_pse_labels.cuda())

            s1_t_imgs = torch.cat((s1_imgs, t_pse_imgs), 0)
            s1_t_labels = torch.cat((s1_labels, t_pse_labels), 0)
            s2_t_imgs = torch.cat((s2_imgs, t_pse_imgs), 0)
            s2_t_labels = torch.cat((s2_labels, t_pse_labels), 0)

            optim_extract.zero_grad()
            optim_s1_cls.zero_grad()
            optim_s2_cls.zero_grad()

            s1_t_feature = extractor(s1_t_imgs)
            s2_t_feature = extractor(s2_t_imgs)
            s1_t_cls = s1_classifier(s1_t_feature)
            s2_t_cls = s2_classifier(s2_t_feature)
            s1_t_cls_loss = get_cls_loss(s1_t_cls, s1_t_labels)
            s2_t_cls_loss = get_cls_loss(s2_t_cls, s2_t_labels)

            torch.autograd.backward([s1_t_cls_loss, s2_t_cls_loss])

            optim_s1_cls.step()
            optim_s2_cls.step()
            optim_extract.step()

        extractor.eval()
        s1_classifier.eval()
        s2_classifier.eval()
        correct = 0
        for (imgs, labels) in target_loader:
            imgs = Variable(imgs.cuda())
            imgs_feature = extractor(imgs)

            s1_cls = s1_classifier(imgs_feature)
            s2_cls = s2_classifier(imgs_feature)
            _,s1_cls = F.softmax(s1_cls)
            _,s2_cls = F.softmax(s2_cls)
            s1_cls = s1_cls.data.cpu().numpy()
            s2_cls = s2_cls.data.cpu().numpy()
            res = s1_cls * s1_weight + s2_cls * s2_weight

            pred = res.argmax(axis=1)
            labels = labels.numpy()
            correct += np.equal(labels, pred).sum()
        current_accuracy = correct * 1.0 / len(target_set)
        print("Current accuracy is: ", current_accuracy)

        if current_accuracy >= max_correct:
            max_correct = current_accuracy
