#%%
"""
Train a diffusion model on images.
"""
import sys
import argparse
import glob, random
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, RandomRotFlip, ToTensor,  Defect, TwoStreamBatchSampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    args = create_argparser().parse_args(args=[])

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()) # script_util 74, unet 393
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000) # uniform

    logger.log("creating data loader...")
    overall_dataset = [file_path for file_path in glob.glob(args.data_dir + '/*.jpg')] 

    train_Hist, test_Hist = train_test_split(overall_dataset, test_size=0.2, random_state=42)


    trainset = Defect(base_dir=train_Hist,
                       split='train',
                       transform =ToTensor())
    labeled_idxs = list(range(args.labelnum)) # [0,1,...,11] # the ratio of labeled data
    unlabeled_idxs = list(range(args.labelnum, args.max_samples)) # [12,9,...,92]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs) #12 12-6

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_datal = DataLoader(trainset,batch_sampler=batch_sampler, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)


    # train_ds = BRATSDataset1(train_Hist, test_flag=False, transform = transforms.Compose([
    #                                                                 RandomRotFlip(),
    #                                                                 ToTensor()]))
    # train_datal= th.utils.data.DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size, # 1
    #     shuffle=True)
    
    val_ds = BRATSDataset(test_Hist, test_flag=True, transform = ToTensor())
    val_datal= th.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True)
        
    data = iter(train_datal)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=train_datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop() # train_util 33行


def create_argparser():
    defaults = dict(
        data_dir="D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=12, #
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100,
        resume_checkpoint='./results/savedmodel040200.pt',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        labelnum=12,
        max_samples = 92,
        seed = 1337,
        labeled_bs = 6

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# %%
