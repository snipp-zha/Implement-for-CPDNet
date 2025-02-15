from __future__ import absolute_import, division, print_function
import cv2
import os
import logging
import torch
import torch.nn as nn
import numpy as np 
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

   
# import argparse

''''
class Logger:
class Parser:
'''
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['name_data'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target, mask=None): #预测结果 真实标签(4,144,240,240)（伪标签）

        pred = F.sigmoid(logits)
        p_flat = pred.view(-1)
        t_flat = target.view(-1)
        inter = p_flat * t_flat
        union = p_flat + t_flat
        if mask is not None:
            mask = mask.view(-1)
            inter = (inter.view(-1) * mask).sum()
            union = (union.view(-1) * mask).sum()
        else:
            # N x C
            inter = inter.view( -1).sum()
            union = union.view(-1).sum()

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
        # pred = F.sigmoid(logits)
        # p_flat = pred.view(-1)
        # t_flat = target.view(-1)
        # intersection = (p_flat * t_flat).sum()

        # dice = (2 * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)

        # return 1 - dice.mean()

class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger
    
def context_mask(img,lab):
    batch_size,  lab_x, lab_y = lab.shape

    final_img = np.zeros_like(img)
    final_lab = np.zeros_like(lab)
    final_labsquare = np.zeros_like(lab)
    define_l = 150

    for i in range(batch_size):
        
        _, thresh = cv2.threshold(lab[i], 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(lab[i])
        # cv2.drawContours(mask, contours, -1, 255, -1)
        if len(contours) > 0:
            mask = np.zeros_like(lab[i])
            cv2.drawContours(mask, contours, -1, 255, -1)
        else:
            print("No contours found")
        # 随机选择图像中的一个区域
        y, x = np.where(mask == 255)
        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
        x1 = x1-10 if x1-10>0 else x1
        y1 = y1-10 if y1-10>0 else y1
        roi_lab = lab[i][y1:y1+define_l, x1:x1+define_l]
        roi_img = img[i][y1:y1+define_l, x1:x1+define_l]
        # 随机缩放和旋转区域
        k = np.random.randint(0, 4)
        roi_lab_transformed = np.rot90(roi_lab, k)
        roi_img_transformed = np.rot90(roi_img, k)
        axis = np.random.randint(0, 2)
        roi_lab_transformed = np.flip(roi_lab_transformed, axis=axis).copy()
        roi_img_transformed = np.flip(roi_img_transformed, axis=axis).copy()

        # # 将变换后的区域复制到图像的其他位置
        x_offset, y_offset = np.random.randint(0, lab[i].shape[1]-roi_lab_transformed.shape[0]), np.random.randint(0, lab[i].shape[0]-roi_lab_transformed.shape[1])
        lab_mask = np.zeros_like(lab[i])
        img_mask = np.zeros_like(img[i])
        img_mask_new = np.zeros_like(img[i])
        lab_mask[x_offset : x_offset+roi_lab_transformed.shape[0], y_offset:y_offset+roi_lab_transformed.shape[1]] = roi_lab_transformed
        img_mask[x_offset : x_offset+roi_img_transformed.shape[0], y_offset:y_offset+roi_img_transformed.shape[1]] = roi_img_transformed
        lab_mask = lab_mask/255 # 将处理后的分割金标准中要交换部分归一化
        img_mask = img_mask/255 # 将处理后的图像中要交换部分归一化
        img_mask_new[img_mask>0]=1 #扩充了边界的模板

        final_img[i] = img_mask
        final_lab[i] = lab_mask
        final_labsquare[i] = img_mask_new

    return final_img, final_lab, final_labsquare

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)