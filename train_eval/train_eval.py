import os
import argparse
import tqdm
import numpy as np
from itertools import chain
from collections import OrderedDict
import torch
import  math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import OfficeImage, LinePlotter
from model import Extractor, Classifier, Discriminator
from model import get_cls_loss, get_dis_loss, get_confusion_loss
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/Users/bytedabce/PycharmProjects/mix_net/data/Office31")
parser.add_argument("-s1", default="amazon")
parser.add_argument("-s2", default="webcam")
parser.add_argument("-t", default="dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--steps", default=8)
parser.add_argument("--snapshot", default="")
parser.add_argument("--s1_weight", default=0.5)
parser.add_argument("--s2_weight", default=0.5)
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--alpha", default=0.8)
parser.add_argument("--gpu_id", default=1)
parser.add_argument("--num_classes", default=31)
parser.add_argument("--threshold", default=0.9)
parser.add_argument("--log_interval", default=5)
parser.add_argument("--cls_epoches", default=10)
parser.add_argument("--gan_epoches", default=5)

args = parser.parse_args()

data_root = args.data_root
batch_size = args.batch_size
shuffle = args.shuffle
num_workers = args.num_workers
steps = args.steps
snapshot = args.snapshot
s1_weight = args.s1_weight
s2_weight = args.s2_weight
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
gpu_id = args.gpu_id
num_classes = args.num_classes
threshold = args.threshold
log_interval = args.log_interval
cls_epoches = args.cls_epoches
gan_epoches = args.gan_epoches
alpha = args.alpha

s1_root = os.path.join(data_root, args.s1, "images")
s1_label = os.path.join(data_root, args.s1, "label.txt")
s2_root = os.path.join(data_root, args.s2, "images")
s2_label = os.path.join(data_root, args.s2, "label.txt")
t_root = os.path.join(data_root, args.t, "images")
t_label = os.path.join(data_root, args.t, "label.txt")
s1_set = OfficeImage(s1_root, s1_label, split="train")
s2_set = OfficeImage(s2_root, s2_label, split="train")
t_set = OfficeImage(t_root, t_label, split="train")
t_set_test = OfficeImage(t_root, t_label, split="test")


s1_loader_raw = torch.utils.data.DataLoader(s1_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
s2_loader_raw = torch.utils.data.DataLoader(s2_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
t_loader_raw = torch.utils.data.DataLoader(t_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)
t_loader_test = torch.utils.data.DataLoader(t_set_test, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)

s1_loader_raw1 = torch.utils.data.DataLoader(s1_set, batch_size=1,
    shuffle=shuffle, pin_memory=True)
s2_loader_raw1 = torch.utils.data.DataLoader(s2_set, batch_size=1,
    shuffle=shuffle, pin_memory=True)
t_loader_raw1 = torch.utils.data.DataLoader(t_set, batch_size=1,
    shuffle=shuffle,pin_memory=True)

extractor = Extractor().cpu()
extractor.load_state_dict(torch.load("/Users/bytedabce/PycharmProjects/mix_net/train_eval/pre_train_model/bvlc_extractor.pth"))
s1_classifier = Classifier(num_classes=num_classes).cpu()
s2_classifier = Classifier(num_classes=num_classes).cpu()
s1_classifier.load_state_dict(torch.load("/Users/bytedabce/PycharmProjects/mix_net/train_eval/pre_train_model/bvlc_s1_cls.pth"))
s2_classifier.load_state_dict(torch.load("/Users/bytedabce/PycharmProjects/mix_net/train_eval/pre_train_model/bvlc_s2_cls.pth"))
s1_t_discriminator = Discriminator().cpu()
s2_t_discriminator = Discriminator().cpu()




def print_log(step, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag, ploter, count):
    print ("Step [%d/%d] Epoch [%d/%d] lr: %f, s1_cls_loss: %.4f, s2_cls_loss: %.4f, s1_t_dis_loss: %.4f, " \
          "s2_t_dis_loss: %.4f, s1_t_confusion_loss_s1: %.4f, s1_t_confusion_loss_t: %.4f, " \
          "s2_t_confusion_loss_s2: %.4f, s2_t_confusion_loss_t: %.4f, selected_source: %s" \
          % (step, steps, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag),
    '''ploter.plot("s1_cls_loss", "train", count, l1)
    ploter.plot("s2_cls_loss", "train", count, l2)
    ploter.plot("s1_t_dis_loss", "train", count, l3)
    ploter.plot("s2_t_dis_loss", "train", count, l4)
    ploter.plot("s1_t_confusion_loss_s1", "train", count, l5)
    ploter.plot("s1_t_confusion_loss_t", "train", count, l6)
    ploter.plot("s2_t_confusion_loss_s2", "train", count, l7)
    ploter.plot("s2_t_confusion_loss_t", "train", count, l8)''')


count = 0
max_correct = 0
max_step = 0
max_epoch = 0
correct = 0
for (imgs, labels) in t_loader_test:
    imgs = Variable(imgs.cpu())
    imgs_feature = extractor(imgs)

    s1_cls = s1_classifier(imgs_feature)
    s2_cls = s2_classifier(imgs_feature)
    s1_cls = F.softmax(s1_cls)
    s2_cls = F.softmax(s2_cls)
    s1_cls = s1_cls.data.cpu().numpy()
    s2_cls = s2_cls.data.cpu().numpy()
    res = s1_cls * s1_weight + s2_cls * s2_weight

    pred = res.argmax(axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()
current_accuracy = correct * 1.0 / len(t_set_test)
print ("Current accuracy is: ", current_accuracy)


ploter = LinePlotter(env_name="bvlc_A_D_2_W")
for step in range(steps):
    # Part 1: assign psudo-labels to t-domain and update the label-dataset
    print ("#################### Part1 ####################")
    extractor.eval()
    s1_classifier.eval()
    s2_classifier.eval()
    
    fin = open(t_label)
    fout = open(os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt"), "w")
    if step > 0:
        s1_weight = s1_weight_loss / (s1_weight_loss + s2_weight_loss)
        s2_weight = s2_weight_loss / (s1_weight_loss + s2_weight_loss)
    print ("s1_weight is: ", s1_weight)
    print ("s2_weight is: ", s2_weight)

    for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(t_loader_test)):
        t_imgs = Variable(t_imgs.cuda())
        t_feature = extractor(t_imgs)
        s1_cls = s1_classifier(t_feature)
        s2_cls = s2_classifier(t_feature)
        s1_cls = F.softmax(s1_cls)
        s2_cls = F.softmax(s2_cls)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()
        
        t_pred = s1_cls * s1_weight + s2_cls * s2_weight
        ids = t_pred.argmax(axis=1)
        for j in range(ids.shape[0]):
            line = fin.readline()
            data = line.strip().split(" ")
            if t_pred[j, ids[j]] >= threshold:
                fout.write(data[0] + " " + str(ids[j]) + "\n")
    fin.close()
    fout.close()
    # Part 2: train F1t, F2t with pseudo labels
    print ("#################### Part2 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    t_pse_label = os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt")
    t_pse_set = OfficeImage(t_root, t_pse_label, split="train")
    t_pse_loader_raw = torch.utils.data.DataLoader(t_pse_set, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
    print ("Length of pseudo-label dataset: ", len(t_pse_set))

    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_cls = optim.Adam(s2_classifier.parameters(), lr=lr, betas=(beta1, beta2))

    for cls_epoch in range(cls_epoches):
        s1_loader, s2_loader, t_pse_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(t_pse_loader_raw)
        for i, (t_pse_imgs, t_pse_labels) in tqdm.tqdm(enumerate(t_pse_loader)):
            try:
                s1_imgs, s1_labels = s1_loader.next()
            except StopIteration:
                s1_loader = iter(s1_loader_raw)
                s1_imgs, s1_labels = s1_loader.next()
            try:
                s2_imgs, s2_labels = s2_loader.next()
            except StopIteration:
                s2_loader = iter(s2_loader_raw)
                s2_imgs, s2_labels = s2_loader.next()
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

            if (i+1) % log_interval == 0:
                print_log(step+1, cls_epoch+1, cls_epoches, lr, s1_t_cls_loss.item(), s2_t_cls_loss.item(), 0, 0, 0, 0, 0, 0, "...", ploter, count)
                count += 1
    
        extractor.eval()
        s1_classifier.eval()
        s2_classifier.eval()
        correct = 0
        for (imgs, labels) in t_loader_test:
            imgs = Variable(imgs.cuda())
            imgs_feature = extractor(imgs)

            s1_cls = s1_classifier(imgs_feature)
            s2_cls = s2_classifier(imgs_feature)
            s1_cls = F.softmax(s1_cls)
            s2_cls = F.softmax(s2_cls)
            s1_cls = s1_cls.data.cpu().numpy()
            s2_cls = s2_cls.data.cpu().numpy()
            res = s1_cls * s1_weight + s2_cls * s2_weight

            pred = res.argmax(axis=1)
            labels = labels.numpy()
            correct += np.equal(labels, pred).sum()
        current_accuracy = correct * 1.0 / len(t_set_test)
        print ("Current accuracy is: ", current_accuracy)

        if current_accuracy >= max_correct:
            max_correct = current_accuracy
            max_step = step
            max_epoch = cls_epoch

       #     torch.save(extractor.state_dict(), os.path.join(snapshot, "p2_extractor_" + str(step) + "_" + str(cls_epoch) + ".pth"))
        #    torch.save(s1_classifier.state_dict(), os.path.join(snapshot, "p2_s1_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
         #   torch.save(s2_classifier.state_dict(), os.path.join(snapshot, "p2_s2_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
    #Part 3: find the nearest samples to fix classifier

    print("#################### Part3 ####################")
    avg_cost = 0
    avg_cost_s1 = 0
    avg_cost_s2 = 0
    a = iter(s1_loader_raw)
    t_imgs_1, t_lable = a.next()
    t_imgs  = Variable(t_imgs_1.cuda())
    t_feature_h = extractor(t_imgs)
    t_sums = t_feature_h-t_feature_h
    s1_sums = t_sums
    s2_sums = t_sums
    for ii, (t_imgs_2, t_labels_2) in tqdm.tqdm(enumerate(t_loader_raw1)):
        t_imgs = Variable(t_imgs_2.cuda())
        t_feature_h = extractor(t_imgs)
        t_sums = t_sums + t_feature_h
    torch.cuda.empty_cache()
    avg_cost = t_sums/len(t_set)
    disc_costs = np.zeros(len(s1_set))
    for j, (s1_img, s1_label) in tqdm.tqdm(enumerate(s1_loader_raw1)):
        img = Variable(s1_img.cuda())
        s1_feature = extractor(img)
        s1_sums = s1_sums+s1_feature
        disc_costs[j] = torch.norm(s1_feature-avg_cost)
    torch.cuda.empty_cache()
    sorted_disc_costs = sorted(disc_costs.tolist(), reverse=False)  # from small to large ones
    threshold_dis = sorted_disc_costs[int(len(s1_set)/4)]
    avg_cost_s1 = s1_sums/len(s1_set)
    for j, (s1_img, s1_label) in tqdm.tqdm(enumerate(s1_loader_raw1)):
        img = Variable(s1_img.cuda())
        label = Variable(s1_label.cuda())
        s1_feature = extractor(img)
        if (torch.norm(s1_feature-avg_cost)) < threshold_dis:
            s1_t_cls = s1_classifier(s1_feature)
            s1_t_cls_loss = get_cls_loss(s1_t_cls, label)
            torch.autograd.backward([s1_t_cls_loss])
            optim_s1_cls.step()
    torch.cuda.empty_cache()
    print("finish s1")
    disc_costs2 = np.zeros(len(s2_set))
    for j, (s2_img, s2_label) in tqdm.tqdm(enumerate(s2_loader_raw1)):
        img = Variable(s2_img.cuda())
        s2_feature = extractor(img)
        disc_costs[j] = torch.norm(s2_feature - avg_cost)
        s2_sums = s2_sums + s2_feature
    torch.cuda.empty_cache()
    sorted_disc_costs2 = sorted(disc_costs2.tolist(), reverse=False)  # from small to large ones
    threshold_dis = sorted_disc_costs2[int(len(s2_set) / 4)]
    avg_cost_s2 = s2_sums / len(s2_set)
    for j, (s2_img, s2_label) in tqdm.tqdm(enumerate(s2_loader_raw1)):
        img = Variable(s2_img.cuda())
        label = Variable(s2_label.cuda())
        s2_feature = extractor(img)
        if (torch.norm(s2_feature - avg_cost)) < threshold_dis:
            s2_t_cls = s2_classifier(s2_feature)
            s2_t_cls_loss = get_cls_loss(s2_t_cls, label)
            torch.autograd.backward([s2_t_cls_loss])
            optim_s2_cls.step()
    torch.cuda.empty_cache()
    print("finish s2")
    correct = 0
    for (imgs, labels) in t_loader_test:
        imgs = Variable(imgs.cuda())
        imgs_feature = extractor(imgs)

        s1_cls = s1_classifier(imgs_feature)
        s2_cls = s2_classifier(imgs_feature)
        s1_cls = F.softmax(s1_cls)
        s2_cls = F.softmax(s2_cls)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()
        res = s1_cls * s1_weight + s2_cls * s2_weight

        pred = res.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
    current_accuracy = correct * 1.0 / len(t_set_test)
    print("Current accuracy is: ", current_accuracy)

    # Part 4: train discriminator and generate mix feature
    print ("#################### Part4 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_t_dis = optim.Adam(s1_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_t_dis = optim.Adam(s2_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    s1_weight_loss = 0
    s2_weight_loss = 0

    for gan_epoch in range(gan_epoches):
        s1_loader, s2_loader, t_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(t_loader_raw)
        for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(t_loader)):
            try:
                s1_imgs, s1_labels = s1_loader.next()
            except StopIteration:
                s1_loader = iter(s1_loader_raw)
                s1_imgs, s1_labels = s1_loader.next()
            try:
                s2_imgs, s2_labels = s2_loader.next()
            except StopIteration:
                s2_loader = iter(s2_loader_raw)
                s2_imgs, s2_labels = s2_loader.next()
            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
            s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
            t_imgs = Variable(t_imgs.cuda())
  
            extractor.zero_grad()
            s1_feature = extractor(s1_imgs)
            s2_feature = extractor(s2_imgs)
            t_feature = extractor(t_imgs)
            s1_cls = s1_classifier(s1_feature)
            s2_cls = s2_classifier(s2_feature)
            s1_t_fake = s1_t_discriminator(s1_feature)
            s1_t_real = s1_t_discriminator(t_feature)
            s2_t_fake = s2_t_discriminator(s2_feature)
            s2_t_real = s2_t_discriminator(t_feature)
            s1_cls_loss = get_cls_loss(s1_cls, s1_labels)
            s2_cls_loss = get_cls_loss(s2_cls, s2_labels)
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            s1_weight_loss += s1_t_dis_loss.data.item()
            s2_weight_loss += s2_t_dis_loss.data.item()

            s1_t_confusion_loss_s1 = get_confusion_loss(s1_t_fake)
            s1_t_confusion_loss_t = get_confusion_loss(s1_t_real)            
            s1_t_confusion_loss = 0.5 * s1_t_confusion_loss_s1 + 0.5 * s1_t_confusion_loss_t

            s2_t_confusion_loss_s2 = get_confusion_loss(s2_t_fake)
            s2_t_confusion_loss_t = get_confusion_loss(s2_t_real)
            s2_t_confusion_loss = 0.5 * s2_t_confusion_loss_s2 + 0.5 * s2_t_confusion_loss_t

            if s1_t_dis_loss.data.item() > s2_t_dis_loss.data.item():
                SELECTIVE_SOURCE = "S1"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss, s1_t_confusion_loss])
            else:
                SELECTIVE_SOURCE = "S2"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss, s2_t_confusion_loss])
            optim_extract.step()

            s1_t_discriminator.zero_grad()
            s2_t_discriminator.zero_grad()
            s1_t_fake = s1_t_discriminator(s1_feature.detach())
            s1_t_real = s1_t_discriminator(t_feature.detach())
            s2_t_fake = s2_t_discriminator(s2_feature.detach())
            s2_t_real = s2_t_discriminator(t_feature.detach())
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            torch.autograd.backward([s1_t_dis_loss, s2_t_dis_loss])
            optim_s1_t_dis.step()
            optim_s2_t_dis.step()

            if (i+1) % log_interval == 0:
                print_log(step+1, gan_epoch+1, gan_epoches, lr, s1_cls_loss.data.item(), s2_cls_loss.data.item(), s1_t_dis_loss.data.item(), \
                          s2_t_dis_loss.data.item(), s1_t_confusion_loss_s1.data.item(), s1_t_confusion_loss_t.data.item(), \
                          s2_t_confusion_loss_s2.data.item(), s2_t_confusion_loss_t.data.item(), SELECTIVE_SOURCE, ploter, count)
                count += 1
        print("calculate weight")
        s1_weight_dis = math.exp(-(torch.mean(avg_cost-avg_cost_s1))**2)
        s2_weight_dis = math.exp(-(torch.mean(avg_cost - avg_cost_s2)) ** 2)
        s1_weight_loss = alpha* (s1_weight_loss/(s1_weight_loss+s2_weight_loss)) +(1-alpha)*(s1_weight_dis/(s1_weight_dis+s2_weight_dis))
        s2_weight_loss = alpha * (s2_weight_loss/(s1_weight_loss+s2_weight_loss)) +(1-alpha)*(s2_weight_dis/(s1_weight_dis+s2_weight_dis))
print("max_correct is :",str(max_correct))
print("max_step is :",str(max_step+1))
print("max_epoch is :",str(max_epoch+1))
ploter.save()
