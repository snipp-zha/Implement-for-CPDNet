#%%
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
from skimage.measure import label
import torch.nn.functional as F
import argparse
import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
import sys, os
import random
sys.path.append(".")
import numpy as np
from guided_diffusion.bratsloader import BRATSDataset, RandomRotFlip, ToTensor, ToTensor1
import torch as th
import torch
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    normalized_img = np.uint8(((img - img.min()) / (img.max() - img.min())) * 255)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask].astype('int64')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def get_cut_mask(out, thres=0.5, nms=0):
    # probs = F.sigmoid(out)
    probs = torch.clip(out,0,1)
    mask = (probs >= thres).type(torch.int64)  # [2, 144, 240, 240]
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
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

def measure_pa_miou(num_class, pre_image, gt_image):
    pre_image = get_cut_mask(pre_image, nms=1)
    pre_image = pre_image.cpu().numpy()
    gt_image = gt_image.cpu().numpy()
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    mIoU = metric.Mean_Intersection_over_Union()
    return mIoU


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
    accuracy = (TP + TN + smooth) / (N + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    iou = (TP + smooth) / (FN + TP + FP + smooth)
    dice = (2 * TP + smooth) / (FN + 2 * TP + FP + smooth)
    F1 = 2 * recall * precision / (recall + precision)
    return accuracy, recall, precision, F1, iou, dice

def main():
    args = create_argparser().parse_args(args=[])
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    overall_dataset = [file_path for file_path in glob.glob(args.data_dir + '/*.jpg')] 

    train_Hist, test_Hist = train_test_split(overall_dataset, test_size=0.2, random_state=42)
  
    train_ds = BRATSDataset(train_Hist, test_flag=False, transform = transforms.Compose([
                                                                    RandomRotFlip(),
                                                                    ToTensor()]))
    train_datal= th.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size, # 1
        shuffle=True)
    
    val_ds = BRATSDataset(test_Hist, test_flag=True, transform = ToTensor1())
    val_datal= th.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    prediction_file = 'D:/HZX/Program files/cache/1761170395/FileRecv/Diffusion-based-Segmentation/emasave_prdiction/prediction'
    origin_file = 'D:/HZX/Program files/cache/1761170395/FileRecv/Diffusion-based-Segmentation/emasave_prdiction/origin'
    label_file = 'D:/HZX/Program files/cache/1761170395/FileRecv/Diffusion-based-Segmentation/emasave_prdiction/label'
    ensemble_file = 'D:/HZX/Program files/cache/1761170395/FileRecv/Diffusion-based-Segmentation/emasave_prdiction/ensemble'
    if not os.path.exists(prediction_file):
        os.makedirs(prediction_file)
    if not os.path.exists(origin_file):
        os.makedirs(origin_file)
    if not os.path.exists(label_file):
        os.makedirs(label_file)
    if not os.path.exists(ensemble_file):
        os.makedirs(ensemble_file)

    acc_val_all = 0
    recall_val_all = 0
    precision_val_all = 0
    F1_val_all = 0
    iou_val_all = 0
    dice_val_all = 0
    miou_val_all = 0

    for step, (b, label)in enumerate(val_datal):
        c = th.randn_like(b[:, :1, ...]) #噪声是单通道的
        img = th.cat((b, c), dim=1)     #add a noise channel$ 5通道 从script_util166行可以知道

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        image_void = torch.zeros_like(c)[0,0]
        tem = torch.zeros_like(c)
        image_void = image_void.cpu().numpy()
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.生成5张独立的预测结果
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known #前者 guass_diffusion 490行
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size), img,  # 3通道有问题
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            ) # 490行

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            print('the shape of sampe is:',sample.shape)
            # s = th.tensor(sample)
            savename = str(step) + '_' + str(i) + '.png'
            tem += sample
            sample_numpy = sample.detach().cpu().numpy()[0][0]
            sample_numpy = visualize(sample_numpy)
            image_void += sample_numpy

            plt.imshow(sample_numpy,cmap='gray')
            plt.axis('off')
            plt.savefig(prediction_file +  '/' + savename, bbox_inches='tight', pad_inches=0)

            b_numpy = b.cpu().numpy()[0][0]
            b_numpy = visualize(b_numpy)
            plt.imshow(b_numpy,cmap='gray')
            plt.axis('off')
            plt.savefig(origin_file +  '/' + savename, bbox_inches='tight', pad_inches=0)

            label_numpy = label.cpu().numpy()[0][0]
            label_numpy = visualize(label_numpy)
            plt.imshow(label_numpy,cmap='gray')
            plt.axis('off')
            plt.savefig(label_file +  '/' + savename, bbox_inches='tight', pad_inches=0)

        accuracy, recall, precision, F1, iou, dice = accuracy_score(tem/args.num_ensemble, label)
        miou = measure_pa_miou(2, tem/args.num_ensemble, label)

        acc_val_all += accuracy.item()
        recall_val_all += recall.item()
        precision_val_all += precision.item()
        F1_val_all += F1.item()
        iou_val_all += iou.item()
        dice_val_all += dice.item()
        miou_val_all += miou.item()
        image_ensemble = image_void / args.num_ensemble
        image_ensemble[image_ensemble >= 128] = 255
        image_ensemble[image_ensemble < 128] = 0
        plt.imshow(image_ensemble, cmap='gray')
        plt.axis('off')
        plt.savefig(ensemble_file + '/' + str(step) + '.png', bbox_inches='tight', pad_inches=0)

    acc_val_avg = acc_val_all / len(val_datal)
    recall_val_avg = recall_val_all / len(val_datal)
    precision_val_avg = precision_val_all / len(val_datal)
    F1_val_avg = F1_val_all / len(val_datal)
    iou_val_avg = iou_val_all / len(val_datal)
    dice_val_avg = dice_val_all / len(val_datal)
    miou_val_avg = miou_val_all / len(val_datal)
    print('accuracy: %.4f, recall: %.4f, precision: %.4f, F1: %.4f, iou: %.4f, dice: %.4f, miou: %.4f' % (
        acc_val_avg, recall_val_avg, precision_val_avg, F1_val_avg, iou_val_avg, dice_val_avg, miou_val_avg))










        #acc_val_avg = acc_val_all / len(val_datal)
        #recall_val_avg = recall_val_all / len(val_datal)
        #precision_val_avg = precision_val_all / len(val_datal)
        #F1_val_avg = F1_val_all / len(val_datal)
        #iou_val_avg = iou_val_all / len(val_datal)
        #dice_val_avg = dice_val_all / len(val_datal)
        #miou_val_avg = miou_val_all / len(val_datal)
        #print('accuracy: %.4f, recall: %.4f, precision: %.4f, F1: %.4f, iou: %.4f, dice: %.4f, miou: %.4f' % (
        #    acc_val_avg, recall_val_avg, precision_val_avg, F1_val_avg, iou_val_avg, dice_val_avg, miou_val_avg))
        #image_ensemble = image_void/args.num_ensemble
        #image_ensemble[image_ensemble>=128]=255
        #image_ensemble[image_ensemble<128]=0
        #plt.imshow(image_ensemble, cmap='gray')
        #plt.axis('off')
        #plt.savefig(ensemble_file +  '/' + str(step) + '.png', bbox_inches='tight', pad_inches=0)

def create_argparser():
    defaults = dict(
        data_dir="D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs",
        clip_denoised=True,
        num_samples=2,
        batch_size=1,
        use_ddim=False,
        model_path="D:/HZX/Program files/cache/1761170395/FileRecv/Diffusion-based-Segmentation/results/savedmodel040200.pt",
        num_ensemble=2      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()



# %%
