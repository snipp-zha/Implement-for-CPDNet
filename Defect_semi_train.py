#%%
from asyncore import write
import model as Model
import os
import sys
import os, time
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage.measure import label
from utils import context_mask
from dataloaders.dataset import *
from utils import mask_DiceLoss, update_ema_variables
from torch.utils.data import DataLoader
from dataloaders.dataset import Defect
from dataset import Dataset, TwoStreamBatchSampler
#from model import ResUnet, init_net, U_Net
#from Models import *
from model import *
from deeplabv3 import build_model
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/ab', help='Name of Dataset') #数据集路径
parser.add_argument('--pre_max_iteration', type=int,  default=5000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=160, help='maximum samples to train') # 115*0.2=92
parser.add_argument('--labeled_bs', type=int, default=6, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu') # 修改
parser.add_argument('--base_lr', type=float,  default=0.05, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=128, help='trained samples') # 修改
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args(args=[]) 

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.sigmoid(out)
    mask = (probs >= thres).type(torch.int64) # [2, 144, 240, 240]
    mask = mask.contiguous()
    if nms == 1:
        mask = LargestCC_pancreas(mask)
    return mask

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path),map_location='cpu')
    net.load_state_dict(state['state_dict'])


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DICE = mask_DiceLoss(nclass=1) # losses第8行

def pre_train(args, snapshot_path):
    DATE_TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime()) #用于格式化时间
    logger_train = SummaryWriter(f'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/semi_visualization/pre_train/{DATE_TIME}/')
    save_path = os.path.join(snapshot_path+f'/{DATE_TIME}/')
    if not os.path.isdir(save_path):
        os.makedirs(save_path) 

    #model = ResU_Net(1)
    model = U_Net(1)
    #model = build_model('Deeplabv3plus_res50', 1, 'resnet34', pretrained=True, out_stride=16, mult_grid=False)
    #model = AttU_Net(1)
    #model = R2U_Net(1)
    #model = R2AttU_Net(1)
    #model = NestedUNet(1)
    #model = FCN(1)
    init_net(model, init_type='normal', init_gain=0.02, device=device)
    overall_dataset = [file_path for file_path in glob.glob('D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/K3/*.jpg')]
    train_data_path, val_data_path = train_test_split(overall_dataset, test_size=0.2, random_state=42) 
    trainset = Defect(base_dir=train_data_path,
                       split='train',
                       transform = RandomRotFlip())
    valset = Defect(base_dir=val_data_path,
                       split='val',
                       transform = ToTensor())
    labelnum = args.labelnum # 12 # 需要根据数据集大小确定 1:9
    labeled_idxs = list(range(labelnum)) # [0,1,...,11] # 需要根据数据集大小调整，有标签占总数据的比例
    unlabeled_idxs = list(range(labelnum, args.max_samples)) # [12,9,...,92]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs) #12 12-6
    # dataset 280行 batch_sampler中4个有标签数据，4个没标签数据
    sub_bs = int(args.labeled_bs/2) # 6/2=3
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # lr 0.01

    model.train()
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_loss = 100
    lr_ = base_lr

    iterator = tqdm(range(pre_max_iterations//2), ncols=70) # 70定义进度条的总长度
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs] #bc=12
            #取出有标签的前半部分 ：0~5 0~11是有标签的 

            volume_batch, label_batch = volume_batch.numpy(), label_batch.numpy()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:] # 把有标签的前半部分再均分成两部分
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:] # 分img_a, img_b是为了两两粘贴复制

            img_b = img_b/255
            lab_b = lab_b/255
            with torch.no_grad():
                final_img, final_lab, final_labsquare = context_mask(img_a,lab_a)
            #
            """Mix Input"""
            volume_batch = final_img* final_labsquare + img_b * (1 - final_labsquare) #两个volumn粘贴复制后合成一个volumn（考虑Batchsize的大小）
            label_batch = final_lab * final_labsquare + lab_b * (1 - final_labsquare) # 4个变成2个了
            volume_batch, label_batch = np.expand_dims(volume_batch, axis=1), np.expand_dims(label_batch, axis=1)
            volume_batch, label_batch = torch.from_numpy(volume_batch).to(device), torch.from_numpy(label_batch).to(device)
            final_labsquare = np.expand_dims(final_labsquare, axis=1)
            final_labsquare = torch.from_numpy(final_labsquare).to(device)            
            outputs = model(volume_batch)
            loss_ce = (F.binary_cross_entropy_with_logits(outputs , label_batch )* final_labsquare).sum() / (final_labsquare.sum() + 1e-16)  + \
                      1/2* (F.binary_cross_entropy_with_logits(outputs , label_batch)* (1 - final_labsquare)).sum() / ((1 - final_labsquare).sum() + 1e-16) 
            loss_dice = DICE(outputs, label_batch, final_labsquare) + \
                        1/2* DICE(outputs, label_batch, (1 - final_labsquare))
            # loss_ce = F.binary_cross_entropy_with_logits(outputs, label_batch)
            # loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss_dice: %03f, loss_ce: %03f'%(iter_num, loss_dice, loss_ce))

            if iter_num % 2500 == 0:
                print('the lr before is:', lr_)
                lr_ = lr_ * 0.1 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('the lr after is:', lr_)

            if iter_num % 100 == 0: #保存模型
                print("===> Validating") #验证模型
                with torch.no_grad():
                    model.eval()
                    loss_all = 0
                    for step, sampled_batch in enumerate(valloader):
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
                        outputs = model(volume_batch)
                        loss_ce = F.binary_cross_entropy_with_logits(outputs, label_batch)
                        loss_dice = DICE(outputs, label_batch)
                        loss = (loss_ce + loss_dice) / 2
                        loss_all += loss
                        volume_batch = volume_batch.detach().cpu().numpy() 
                        label_batch = label_batch.detach().cpu().numpy()
                        label_map = get_cut_mask(outputs, nms=1).detach().cpu().numpy()
                        logger_train.add_image('inputs', volume_batch[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step) # 只反复batch最后一个
                        logger_train.add_image('labels', label_batch[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step)
                        logger_train.add_image('outputs', label_map[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step)
                    loss_val_avg = loss_all/len(valloader)

                if loss_val_avg < best_loss:
                    print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, loss_val_avg))
                    best_loss = loss_val_avg
                    torch.save({
                        'iter_num': iter_num,
                        'state_dict' : model.state_dict()
                    },os.path.join(save_path, "the_best_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                    print("Saving model at",os.path.join('D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/checkpoint/', "the_best_model.pth"))

                if iter_num == pre_max_iterations:
                    torch.save({
                        'iter_num': iter_num,
                        'state_dict' : model.state_dict()
                    },os.path.join(save_path, "the_last_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                    print("Saving model at",os.path.join('D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/checkpoint/', "the_last_model.pth"))
                model.train()

            if iter_num >= pre_max_iterations: # 2000
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break

def self_train(args, pre_snapshot_path, self_snapshot_path):

    DATE_TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime()) #用于格式化时间
    logger_train = SummaryWriter(f'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/semi_visualization/self_train/{DATE_TIME}/')
    save_path = os.path.join(self_snapshot_path+f'/{DATE_TIME}/')
    if not os.path.isdir(save_path):
        os.makedirs(save_path) 
        
    model = U_Net(1)
    ema_model = U_Net(1)
    #model = build_model('Deeplabv3plus_res50', 1, 'resnet34', pretrained=True, out_stride=16, mult_grid=False)
    #ema_model = build_model('Deeplabv3plus_res50', 1, 'resnet34', pretrained=True, out_stride=16, mult_grid=False)
    #model = AttU_Net(1)
    #ema_model = AttU_Net(1)
    #model = NestedUNet(1)
    #ema_model = NestedUNet(1)
    #model = FCN(1)
    #ema_model = FCN(1)
    #model = R2U_Net(1)
    #ema_model = R2U_Net(1)
    #model = R2AttU_Net(1)
    #ema_model = R2AttU_Net(1)
    for param in ema_model.parameters():
            param.detach_()   # ema_model set 不需要计算梯度
    
    overall_dataset = [file_path for file_path in glob.glob('D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/K3/*.jpg')]
    train_data_path, val_data_path = train_test_split(overall_dataset, test_size=0.2, random_state=42) 
    trainset = Defect(base_dir=train_data_path,
                       split='train',
                       transform = RandomRotFlip())
    valset = Defect(base_dir=val_data_path,
                       split='val',
                       transform = ToTensor())
    
    labelnum = args.labelnum # 12 # 需要根据数据集大小确定 1:9
    labeled_idxs = list(range(labelnum)) # [0,1,...,11] # 需要根据数据集大小调整，有标签占总数据的比例
    unlabeled_idxs = list(range(labelnum, args.max_samples)) # [12,9,...,92]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs) #12 12-6
    # dataset 280行 batch_sampler中4个有标签数据，4个没标签数据
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # lr 0.01 

    pretrained_model = 'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_128_labeled/pre_train/20231109-233243/the_best_model_228_0.2288.pth' # 加载的参数可以是不同轮数的
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model) #两个模型加载了一样的权重

    model = model.to(device)
    ema_model = ema_model.to(device)

    model.train()
    ema_model.train() # 教师模型

    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_loss = 100

    lr_ = base_lr
    iterator = tqdm(range(self_max_iterations//2), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader): # batchsize=12
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.numpy(), label_batch.numpy()

            img_a, lab_a = volume_batch[:args.labeled_bs], label_batch[:args.labeled_bs] # 6

            unimg_b = volume_batch[args.labeled_bs:] #无标签数据
            unimg_b = np.expand_dims(unimg_b/255, axis=1)
            unimg_b = torch.from_numpy(unimg_b).to(device)  
       
            with torch.no_grad():
                unoutput_b = ema_model(unimg_b) # 先对无标签数据生成标签
                plab_b = get_cut_mask(unoutput_b, nms=1) # 大于0.5的地方和最大连通域的地方保留
                final_img, final_lab, final_labsquare = context_mask(img_a,lab_a)

            final_img, final_lab, final_labsquare = np.expand_dims(final_img, axis=1), np.expand_dims(final_lab, axis=1), np.expand_dims(final_labsquare, axis=1)
            final_img, final_lab, final_labsquare = torch.from_numpy(final_img).to(device), torch.from_numpy(final_lab).to(device), torch.from_numpy(final_labsquare).to(device)
            
            volume_batch = final_img * final_labsquare + unimg_b * (1 - final_labsquare) #两个volumn粘贴复制后合成一个volumn（考虑Batchsize的大小）
            label_batch = final_lab  * final_labsquare + plab_b * (1 - final_labsquare) # 4个变成2个了
            # volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs= model(volume_batch)
            loss_ce = (F.binary_cross_entropy_with_logits(outputs , label_batch )* final_labsquare).sum() / (final_labsquare.sum() + 1e-16)  + \
                      1/2* (F.binary_cross_entropy_with_logits(outputs , label_batch)* (1 - final_labsquare)).sum() / ((1 - final_labsquare).sum() + 1e-16) 
            loss_dice = DICE(outputs, label_batch, final_labsquare) + \
                        1/2* DICE(outputs, label_batch, (1 - final_labsquare))
            # loss_ce = F.binary_cross_entropy_with_logits(outputs, label_batch)
            # loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss_dice: %03f, loss_ce: %03f'%(iter_num, loss_dice, loss_ce))

            update_ema_variables(model, ema_model, 0.99) # 通过学生模型的参数更新教师模型的参数

             # change lr
            if iter_num % 5000 == 0:
                print('the lr before is:', lr_)
                lr_ = lr_ * 0.1  # 原本是2500
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('the lr after is:', lr_)

            if iter_num % 100 == 0:

                print("===> Validating") #验证模型
                with torch.no_grad():
                    model.eval()
                    loss_all = 0
                    for step, sampled_batch in enumerate(valloader):
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
                        outputs = model(volume_batch)
                        loss_ce = F.binary_cross_entropy_with_logits(outputs, label_batch)
                        loss_dice = DICE(outputs, label_batch)
                        loss = (loss_ce + loss_dice) / 2
                        loss_all += loss
                        volume_batch = volume_batch.detach().cpu().numpy() 
                        label_batch = label_batch.detach().cpu().numpy()
                        label_map = get_cut_mask(outputs, nms=1).detach().cpu().numpy()
                        logger_train.add_image('inputs', volume_batch[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step) # 只反复batch最后一个
                        logger_train.add_image('labels', label_batch[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step)
                        logger_train.add_image('outputs', label_map[0,0], dataformats='HW', global_step=(iter_num//100)*len(valloader)+step)
                    loss_val_avg = loss_all/len(valloader)

                if loss_val_avg < best_loss:
                    print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, loss_val_avg))
                    best_loss = loss_val_avg
                    torch.save({
                        'iter_num': iter_num,
                        'state_dict' : model.state_dict()
                    },os.path.join(save_path, "the_best_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                    print("Saving model at",os.path.join('D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/leo_segmentation/checkpoint/', "the_best_model.pth"))

                if iter_num == self_max_iterations:
                    torch.save({
                        'iter_num': iter_num,
                        'state_dict' : model.state_dict()
                    },os.path.join(save_path, "the_last_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                    print("Saving model at",os.path.join('D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/leo_segmentation/checkpoint/', "the_last_model.pth"))

                model.train()
            
            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_{}_labeled/pre_train".format(args.labelnum) # 12
    self_snapshot_path = "D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_{}_labeled/self_train".format(args.labelnum)
    print("Strating Industrial defect training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code') # 递归删除文件夹下的所有子文件夹和子文件

    # -- Pre-Training
    #logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #logging.info(str(args))
    #pre_train(args, pre_snapshot_path) # 118行
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
# %%
