import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from util import OfficeImage, weights_init, print_args
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root1", default="/Users/bytedabce/PycharmProjects/mix_net/data/Office31/amazon")
parser.add_argument("--data_root2", default="/Users/bytedabce/PycharmProjects/mix_net/data/Office31/webcam")
parser.add_argument("--source", default="")
parser.add_argument("--target", default="/Users/bytedabce/PycharmProjects/mix_net/data/Office31/dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--pre_epoches", default=1, type=int)
parser.add_argument("--epoch", default=1, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--class_num", default=31)
parser.add_argument("--extract", default=True)
parser.add_argument("--radius", default=25.0)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

source_root1 = os.path.join(args.data_root1, args.source, "images")
source_label1 = os.path.join(args.data_root1, args.source, "label.txt")
source_root2 = os.path.join(args.data_root2, args.source, "images")
source_label2 = os.path.join(args.data_root2, args.source, "label.txt")

target_root = os.path.join(args.source, args.target, "images")
target_label = os.path.join(args.source, args.target, "label.txt")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

source_set1 = OfficeImage(source_root1, source_label1, train_transform)
source_set2 = OfficeImage(source_root2, source_label2, train_transform)

target_set = OfficeImage(target_root, target_label, train_transform)

source_loader1 = torch.utils.data.DataLoader(source_set1, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
source_loader2 = torch.utils.data.DataLoader(source_set2, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG1 = ResBase50().cpu()
netF1 = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cpu()
netF1.apply(weights_init)

netG2 = ResBase50().cpu()
netF2 = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cpu()
netF2.apply(weights_init)

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - args.radius) ** 2
    return args.weight_L2norm * l
print("step 1")
opt_g1 = optim.SGD(netG1.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f1 = optim.SGD(netF1.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
opt_g2 = optim.SGD(netG2.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f2 = optim.SGD(netF2.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

print("step_2")
for epoch in range(1, args.pre_epoches + 1):
    for i, (s_imgs, s_labels) in tqdm.tqdm(enumerate(source_loader1)):
        if s_imgs.size(0) != args.batch_size:
            continue
            
        s_imgs = Variable(s_imgs.cpu())
        s_labels = Variable(s_labels.cpu())
        print("epoch:"+str(epoch))
        opt_g1.zero_grad()
        opt_f1.zero_grad()
        s_bottleneck = netG1(s_imgs)
        s_fc2_emb, s_logit = netF1(s_bottleneck)
        s_fc2_ring_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        s_cls_loss = get_cls_loss(s_logit, s_labels)

        loss = s_cls_loss + s_fc2_ring_loss 
        loss.backward()

        opt_g1.step()
        opt_f1.step()
print("finish data1 step2")
for epoch in range(1, args.pre_epoches + 1):
    for i, (s_imgs, s_labels) in tqdm.tqdm(enumerate(source_loader2)):
        if s_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cpu())
        s_labels = Variable(s_labels.cpu())
        print("epoch:" + str(epoch))
        opt_g2.zero_grad()
        opt_f2.zero_grad()
        s_bottleneck = netG2(s_imgs)
        s_fc2_emb, s_logit = netF2(s_bottleneck)
        s_fc2_ring_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        s_cls_loss = get_cls_loss(s_logit, s_labels)

        loss = s_cls_loss + s_fc2_ring_loss
        loss.backward()

        opt_g2.step()
        opt_f2.step()
print("finish data2 step2")

print("step_3")
for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader1)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))


    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader1)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cpu())
        s_labels = Variable(s_labels.cpu())
        t_imgs = Variable(t_imgs.cpu())
        
        opt_g1.zero_grad()
        opt_f1.zero_grad()

        s_bottleneck = netG1(s_imgs)
        t_bottleneck = netG1(t_imgs)
        s_fc2_emb, s_logit = netF1(s_bottleneck)
        t_fc2_emb, t_logit = netF1(t_bottleneck)

        s_cls_loss = get_cls_loss(s_logit, s_labels)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

        loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        loss.backward()

        opt_g1.step()
        opt_f1.step()
    if epoch % 10 == 1:
        torch.save(netG1.state_dict(), os.path.join(args.snapshot, "Office31_HAFN_1_" + args.task + "_netG_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth"))
        torch.save(netF1.state_dict(), os.path.join(args.snapshot, "Office31_HAFN_1_" + args.task + "_netF_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth"))

print("finish data1 step3")
for epoch in range(1, args.epoch + 1):
    source_loader_iter = iter(source_loader2)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))

    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader2)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cpu())
        s_labels = Variable(s_labels.cpu())
        t_imgs = Variable(t_imgs.cpu())

        opt_g1.zero_grad()
        opt_f1.zero_grad()

        s_bottleneck = netG2(s_imgs)
        t_bottleneck = netG2(t_imgs)
        s_fc2_emb, s_logit = netF2(s_bottleneck)
        t_fc2_emb, t_logit = netF2(t_bottleneck)

        s_cls_loss = get_cls_loss(s_logit, s_labels)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

        loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        loss.backward()

        opt_g2.step()
        opt_f2.step()
    if epoch % 10 == 1:
        torch.save(netG1.state_dict(), os.path.join(args.snapshot,"Office31_HAFN_2_" + args.task + "_netG_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth"))
        torch.save(netF1.state_dict(), os.path.join(args.snapshot, "Office31_HAFN_2_" + args.task + "_netF_" + args.post + "." + args.repeat + "_" + str(epoch) + ".pth"))