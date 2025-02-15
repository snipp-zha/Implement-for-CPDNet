import random
import torch
import torch.nn as nn
#from model import ResUnet, init_net, U_Net
from torch.utils.data import DataLoader
from dataloaders.dataset import Defect
from dataset import Dataset, TwoStreamBatchSampler
import glob
#from model import *
#from networks.unet import *
from Models import *
from sklearn.model_selection import train_test_split
from dataloaders.dataset import *
from deeplabv3 import build_model
import torch.nn.functional as F
from skimage.measure import label
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def worker_init_fn(worker_id):
    random.seed(1337+worker_id)

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
        labels = label[n_prob]
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = torch.sum((prediction == 1) & (groundtruth == 0))
    FN = torch.sum((prediction == 0) & (groundtruth == 1))
    TP = torch.sum((prediction == 1) & (groundtruth == 1))
    TN = torch.sum((prediction == 0) & (groundtruth == 0))

    return FP, FN, TP, TN
    
def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""
    prediction = get_cut_mask(prediction, nms=1)
    groundtruth = groundtruth > 0.5
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    smooth = 1e-6
    accuracy = (TP + TN + smooth)/ (N + smooth)
    recall = (TP + smooth)/(TP + FN + smooth)
    precision = (TP + smooth)/(TP + FP + smooth)
    iou = (TP + smooth)/(FN + TP + FP + smooth)
    dice = (2*TP + smooth)/(FN + 2*TP + FP + smooth)
    F1 = 2*recall*precision/(recall+precision)
    return accuracy, recall, precision, F1, iou, dice

def load_net(net, path):
   state = torch.load(str(path),map_location='cpu')
   net.load_state_dict(state['state_dict'])
#def load_net(net, path):
 #   state = torch.load(str(path),map_location='cpu')
  #  net.load_state_dict(torch.load(path))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# overall_dataset = [file_path for file_path in glob.glob('/opt/data/private/leo_dataset/磁瓦缺陷数据集/MT_Blowhole/Imgs/*.jpg')] 
# train_data_path, val_data_path = train_test_split(overall_dataset, test_size=0.2, random_state=42) 
# valset = Defect(base_dir=val_data_path,
#                     split='val',
#                     transform = ToTensor())    
# valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
image_path = 'D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Break/Imgs/exp5_num_284552.jpg'
label_path = image_path.replace('jpg','png')
image = np.array(Image.open(image_path), dtype='float32') 
image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA) 
label = np.array(Image.open(label_path), dtype='uint8') 
label = cv2.resize(label, (256,256), interpolation=cv2.INTER_AREA) 
label[label>=128]=255
label[label<128]=0

image = abs(image/255)
print(image)
label = abs(label/255)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

#model = UNet(1,2)
#model = build_model('Deeplabv3plus_res50', 1, 'resnet34', pretrained=True, out_stride=16, mult_grid=False)
model = FCN(1)
#model = ResU_Net(1)
#model = U_Net(1)
pretrained_model = 'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_9_labeled/pre_train/20231117-155127FCN/the_best_model_1999_0.5859.pth'
load_net(model, pretrained_model)

model = model.to(device)

with torch.no_grad():
    model.eval()

    image= image.to(device)
    output = abs(model(image))
    print(output)
    #output = get_cut_mask(output, nms=1)
    image = image.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    plt.figure(dpi=200)
    #plt.subplot(131)
    #plt.imshow(image[0,0])
    #plt.axis('off')
    #ax = plt.gca()
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    plt.imshow(output[0,0],cmap="gray")
    plt.axis('off')
    plt.savefig("a.png", bbox_inches='tight', pad_inches=0)


    #plt.subplot(133)
    #plt.imshow(label)
    #plt.axis('off')
    plt.show()

    # acc_val_all = 0
    # recall_val_all = 0
    # precision_val_all = 0
    # F1_val_all = 0
    # iou_val_all = 0
    # dice_val_all = 0

    # for step, sampled_batch in enumerate(valloader):
    #     volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    #     print('input:',torch.max(volume_batch),torch.min(volume_batch))
    #     print('label:',torch.max(label_batch),torch.min(label_batch))
    #     volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
    #     outputs = model(volume_batch)
    #     accuracy, recall, precision, F1, iou, dice = accuracy_score(outputs, label_batch)

    #     acc_val_all += accuracy.item()
    #     recall_val_all += recall.item()
    #     precision_val_all += precision.item()
    #     F1_val_all += F1.item()
    #     iou_val_all += iou.item()
    #     dice_val_all += dice.item()

    # acc_val_avg =  acc_val_all / len(valloader) 
    # recall_val_avg =  recall_val_all / len(valloader)  
    # precision_val_avg =  precision_val_all / len(valloader)  
    # F1_val_avg =  F1_val_all / len(valloader)  
    # iou_val_avg =  iou_val_all / len(valloader)  
    # dice_val_avg =  dice_val_all / len(valloader)
    # print('accuracy: %.4f, recall: %.4f, precision: %.4f, F1: %.4f, iou: %.4f, dice: %.4f'% (acc_val_avg, recall_val_avg, precision_val_avg, F1_val_avg, iou_val_avg, dice_val_avg))
