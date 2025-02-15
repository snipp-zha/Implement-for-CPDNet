#%%
from model import *
from dataset import *
import time 
import torch
import torch.nn as nn
import glob
from random import randint
import os
from PIL import Image
from dataset import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loss import image_compare_loss
from model import ResUnet, init_net


class Train:
    def __init__(self, args):

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.lr= args.lr



        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self):

        num_epoch = self.num_epoch
        lr = self.lr
        batch_size = self.batch_size
        device = self.device

        All_Images = [file_path for file_path in glob.glob('/opt/data/private/DIV2K/DIV2K_train_HR/*')] 
        print(len(All_Images))
        Images_train, Images_val = train_test_split(All_Images, test_size=0.1, random_state=42)
        train_array = []
        for data_path in Images_train:
            origin_image = Image.open(data_path).convert('RGB')
            origin_image = np.asarray(origin_image).astype(np.float32)
            origin_image = (origin_image-np.min(origin_image))/(np.max(origin_image)-np.min(origin_image)) # (3, 2970, 3660)
            for i in range(6):
                begin = randint(0, min(origin_image.shape[0],origin_image.shape[1])-256)
                image = origin_image[begin:begin+256,begin:begin+256,0]
                train_array.append(image)


        val_array = []
        for data_path in Images_val:    
            origin_image = Image.open(data_path).convert('RGB')
            origin_image = np.asarray(origin_image).astype(np.float32)
            origin_image = (origin_image-np.min(origin_image))/(np.max(origin_image)-np.min(origin_image)) # (3, 2970, 3660)
            for i in range(6):
                begin = randint(0, min(origin_image.shape[0],origin_image.shape[1])-256)
                image = origin_image[begin:begin+256,begin:begin+256,0]
                val_array.append(image)
        #64 64 1
        dataset_train = Dataset(train_array)
        dataset_val = Dataset(val_array)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        model = ResUnet(4) 
        init_net(model, init_type='normal', init_gain=0.02, device=device)

        optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999)) 
        lr_scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(optim, [50000, 100000, 200000, 300000])
        ## load from checkpoints
        st_epoch = 0
        best_loss = 100
        ## setup tensorboard
        DATE_TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime()) #用于格式化时间
        logger_train = SummaryWriter(f'/opt/data/private/leo_segmentation/log_prediction/{DATE_TIME}/train/')
        logger_val = SummaryWriter(f'/opt/data/private/leo_segmentation/log_prediction/{DATE_TIME}/val/')
        if not os.path.isdir(f'/opt/data/private/leo_segmentation/checkpoint/{DATE_TIME}/'):
            os.makedirs(f'/opt/data/private/leo_segmentation/checkpoint/{DATE_TIME}/') 

        for epoch in range(st_epoch + 1, num_epoch + 1):
            model.train()
            num_epoch_no_improvement = 0
            loss_train = 0
            print('------{}------'.format(epoch))
            for batch, (input, label) in enumerate(loader_train, 1):
                print('------{}/{}------'.format(batch, num_train))

                input = input.to(device)
                label = label.to(device)
                B, _, H, W = input.shape
                mask = (nn.init.uniform_(torch.zeros(B, 1, H, W, device=input.device)) > 0.3).float() # T x N x H x W
                input_img = input * mask # T x N x H x W 置为0的地方就被屏蔽了

                # forward model
                output = model(input_img)
                # backward model
                optim.zero_grad()

                loss = image_compare_loss(output, label)
                loss.backward()
                loss_train += loss.item()
        
                lr_scheduler_generator.step()
                optim.step()

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'% (epoch, batch, num_train, loss.item()))


            ## show output
            input_img = input_img.detach().cpu().numpy()[0][0]
            label = label.detach().cpu().numpy()[0][0]
            output = output.detach().cpu().numpy()[0][0]

            input_img = np.clip(input_img, 0, 1)
            label = np.clip(label, 0, 1)
            output = np.clip(output, 0, 1)
            dif = np.clip(abs(label - output), 0, 1)

            loss_train_avg = loss_train / num_train

            logger_train.add_scalar('loss', np.mean(loss_train_avg), epoch)
            logger_train.add_images('input_img', input_img, epoch, dataformats='HW')
            logger_train.add_images('output', output, epoch, dataformats='HW')
            logger_train.add_images('label', label, epoch, dataformats='HW')
            logger_train.add_images('dif', dif, epoch, dataformats='HW')


            ## validation phase
            print("------validating------")
            with torch.no_grad():
                model.eval()

                loss_val = 0

                for batch, (input, label) in enumerate(loader_val, 1):
                    print('------{}/{}------'.format(batch,num_val))

                    input = input.to(device)
                    label = label.to(device)
                    B, _, H, W = input.shape
                    mask = (nn.init.uniform_(torch.zeros(B, 1, H, W, device=input.device)) > 0.3).float() # T x N x H x W
                    input_img = input * mask # T x N x H x W 置为0的地方就被屏蔽了

                    # forward netG
                    output = model(input_img)

                    loss = image_compare_loss(output, label)
                    loss_val += loss.item()
                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'% (epoch, batch, num_val, loss.item()))


                input_img = input_img.detach().cpu().numpy()[0][0]
                label = label.detach().cpu().numpy()[0][0]
                output = output.detach().cpu().numpy()[0][0]

                input_img = np.clip(input_img, 0, 1)
                label = np.clip(label, 0, 1)
                output = np.clip(output, 0, 1)
                dif = np.clip(abs(label - output), 0, 1)

                loss_val_avg = loss_val / num_val

                logger_val.add_scalar('loss', np.mean(loss_val_avg), epoch)     
                logger_val.add_images('input_img', input_img, epoch, dataformats='HW')
                logger_val.add_images('output', output, epoch, dataformats='HW')
                logger_val.add_images('label', label, epoch, dataformats='HW')
                logger_val.add_images('dif', dif, epoch, dataformats='HW')

            if loss_val_avg < best_loss:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, loss_val_avg))
                best_loss = loss_val_avg
                torch.save({
                    'epoch': epoch,
                    'state_dict' : model.state_dict()
                },os.path.join(f'/opt/data/private/leo_segmentation/checkpoint/{DATE_TIME}/', "the_best_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                print("Saving model at",os.path.join('/opt/data/privateleo_segmentation/checkpoint/', "the_best_model.pth"))
            else:
                print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
                num_epoch_no_improvement += 1  
            if  num_epoch_no_improvement == 30:
                torch.save({
                    'epoch': epoch,
                    'state_dict' : model.state_dict()
                },os.path.join(f'/opt/data/private/leo_segmentation/checkpoint/{DATE_TIME}/', "the_earlystop_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                print("Saving model at",os.path.join('/opt/data/private/leo_segmentation/checkpoint/', "early stopping"))
                break
            if epoch == 300:
                torch.save({
                    'epoch': epoch,
                    'state_dict' : model.state_dict()
                },os.path.join(f'/opt/data/private/leo_segmentation/checkpoint/{DATE_TIME}/', "the_last_model_{:03d}_{:.4f}.pth".format(epoch, loss_val_avg)))
                print("Saving model at",os.path.join('/opt/data/private/leo_segmentation/checkpoint/', "the_last_model.pth"))  



# %%
