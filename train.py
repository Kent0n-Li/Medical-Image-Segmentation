import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.UNet import U_Net, R2U_Net, AttU_Net, NestedUNet
from networks.YourNet import Your_Net
from networks.GT_UNet import GT_U_Net
from networks.model.BiSeNet import BiSeNet
from networks.model.BiSeNetV2 import BiSeNetV2
from networks.model.DDRNet import DDRNet
from networks.model.DeeplabV3Plus import Deeplabv3plus_res50
from networks.model.FCN_ResNet import FCN_ResNet
from networks.model.HRNet import HighResolutionNet
from networks.SegNet import SegNet


import torch.utils.data as data
import json
import cv2
from PIL import Image
import SimpleITK as sitk
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
import torch.nn as nn
import torch.optim as optim
from networks.swin_config import get_swin_config
import requests
import matplotlib.pyplot as plt
import shutil
from logHelper import setup_logger
from config import output_file, parse_args
import albumentations as A

args = parse_args(type='train')


def calculate_dice(pred, gt):
    # 用于计算 Dice 相似系数
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0  # 如果两者都是空集，Dice 应为 1

    return 2 * intersection / union


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_dice(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    else:
        return 0


def download_model(url, destination):
    chunk_size = 8192  # Size of each chunk in bytes

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("Weights downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)


class DynamicDataset(data.Dataset):
    def __init__(self, img_path, gt_path, size=None, augmentations=None):

        self.img_name = os.listdir(img_path)
        self.size = size
        self.img_path = img_path
        self.gt_path = gt_path
        self.augmentations = augmentations

    def __getitem__(self, item):
        imagename = self.img_name[item]
        img_path = os.path.join(self.img_path, imagename)
        gt_path = os.path.join(self.gt_path, imagename)

        file_end = imagename.split('.')[-1]

        if file_end in ['png']:
            npimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            npimg = np.array(npimg)
            npgt = Image.open(gt_path)
            npgt = np.array(npgt)

        elif file_end in ['gz', 'nii.gz', 'nii']:
            npimg = sitk.ReadImage(img_path)
            npimg = sitk.GetArrayFromImage(npimg)
            npgt = sitk.ReadImage(gt_path)
            npgt = sitk.GetArrayFromImage(npgt)

        if self.augmentations:
            augmented = self.augmentations(image=npimg, mask=npgt)
            npimg = augmented['image'].copy()
            npgt = augmented['mask'].copy()


        if npimg.ndim == 2:
            npimg = np.expand_dims(npimg, axis=0)
        elif npimg.ndim == 3:
            npimg = npimg.transpose((2, 0, 1))

        ori_shape = npimg.shape
        npgt = np.expand_dims(npgt, axis=0)

        npimg = torch.from_numpy(npimg)
        npgt = torch.from_numpy(npgt)

        if self.size is not None:
            resize = transforms.Resize(size=(self.size, self.size), antialias=None)
            npimg = resize(npimg)
            npgt = resize(npgt)
        else:
            adapt_size = transforms.Resize(size=(int(npimg.shape[1] / 64 + 1) * 64, int(npimg.shape[2] / 64 + 1) * 64),
                                           antialias=None)
            npimg = adapt_size(npimg)
            npgt = adapt_size(npgt)


        npgt = torch.squeeze(npgt)
        return npimg, npgt, imagename, ori_shape

    def __len__(self):
        size = int(len(self.img_name))
        return size


def format_progress_message(iteration, total):

    progress_percentage = (iteration / total) * 100
    progress_bar_length = int(50 * iteration // total)
    progress_bar = '#' * progress_bar_length + '-' * (50 - progress_bar_length)

    return f"{progress_percentage:3.0f}%|{progress_bar}| {iteration}/{total}"



def train(batch_size=4, max_epochs=200, base_lr=0.01, seed=1234, n_gpu=1, model_name='swinunet'):

    data_json_file = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'dataset.json')

    with open(data_json_file) as f:
        json_data = json.load(f)
        num_classes = json_data['label_class_num']
        in_channels = json_data['img_channel']
        args.img_size = json_data['imgae_size']

        #data augmentation
        #        body: JSON.stringify({ Blur:Blur, blur_limit_min_Blur:blur_limit_min_Blur, blur_limit_max_Blur:blur_limit_max_Blur, RandomBrightnessContrast:RandomBrightnessContrast, brightness_limit_min:brightness_limit_min, brightness_limit_max:brightness_limit_max, contrast_limit_min:contrast_limit_min, contrast_limit_max:contrast_limit_max, RandomRotate90:RandomRotate90, VerticalFlip:VerticalFlip, HorizontalFlip:HorizontalFlip, dataset:dataset })


        RandomBrightnessContrast = json_data['RandomBrightnessContrast']
        brightness_limit_min = json_data['brightness_limit_min']
        brightness_limit_max = json_data['brightness_limit_max']
        contrast_limit_min = json_data['contrast_limit_min']
        contrast_limit_max = json_data['contrast_limit_max']
        RandomRotate90 = json_data['RandomRotate90']
        VerticalFlip = json_data['VerticalFlip']
        HorizontalFlip = json_data['HorizontalFlip']


                            
    augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=(brightness_limit_min, brightness_limit_max),
                                       contrast_limit=(contrast_limit_min, contrast_limit_max)) if RandomBrightnessContrast else A.NoOp(),
            A.RandomRotate90(p=0.5) if RandomRotate90 else A.NoOp(),
            A.VerticalFlip(p=0.5) if VerticalFlip else A.NoOp(),
            A.HorizontalFlip(p=0.5) if HorizontalFlip else A.NoOp()
        ])



    args.seed = seed
    args.max_epochs = max_epochs
    args.base_lr = base_lr
    args.batch_size = batch_size
    args.n_gpu = n_gpu

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    output_folder = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'])
    os.makedirs(output_folder, exist_ok=True)
    

    imageTr_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTr')
    labelTr_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTr')
    imageVal_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesVal')
    labelVal_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsVal')
    imageTs_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs')
    labelTs_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs')



    model_name = os.environ['MODEL_NAME']
    print(model_name)

    if model_name == 'unet':
        model = U_Net(in_ch=in_channels, out_ch=num_classes).to(device)

    elif model_name == 'transunet':
        download_url = 'https://huggingface.co/kenton-li/nnSAM/resolve/main/R50%2BViT-B_16.npz'
        vit_name = 'R50-ViT-B_16'
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_classes
        args.img_size = 224
        args.vit_patches_size = 16
        config_vit.n_skip = 3
        config_vit.pretrained_path = './networks/R50+ViT-B_16.npz'
        if not os.path.exists(config_vit.pretrained_path):
            download_model(download_url, config_vit.pretrained_path)
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=224, num_classes=num_classes).to(device)

        model.load_from(weights=np.load('networks/R50+ViT-B_16.npz'))

    elif model_name == 'swinunet':
        args.img_size = 224
        args.cfg = './networks/swin_tiny_patch4_window7_224_lite.yaml'
        args.opts = None
        swin_config = get_swin_config(args)
        model = SwinUnet(swin_config, img_size=224, num_classes=num_classes).to(device)
        # url = "https://drive.google.com/uc?id=1TyMf0_uvaxyacMmVzRfqvLLAWSOE2bJR"
        url = 'https://huggingface.co/kenton-li/nnSAM/resolve/main/swin_tiny_patch4_window7_224.pth'
        output = swin_config.MODEL.PRETRAIN_CKPT
        if not os.path.exists(output):
            download_model(url, output)
            # gdown.download(url, output, quiet=False)
        model.load_from(swin_config)

    elif model_name == 'unetpp':
        model = NestedUNet(in_ch=in_channels, out_ch=num_classes).to(device)

    elif model_name == 'attunet':
        model = AttU_Net(in_ch=in_channels, out_ch=num_classes).to(device)

    elif model_name == 'r2unet':
        model = R2U_Net(in_ch=in_channels, out_ch=num_classes).to(device)

    elif model_name == 'gtunet':
        model = GT_U_Net(in_ch=in_channels, out_ch=num_classes).to(device)
        args.img_size = 256

    elif model_name == 'bisenet':
        model = BiSeNet(in_ch=in_channels, out_ch=num_classes).to(device)
    
    elif model_name == 'ddrnet':
        model = DDRNet(in_ch=in_channels, out_ch=num_classes).to(device)
    
    elif model_name == 'deeplabv3plus':
        model = Deeplabv3plus_res50(in_ch=in_channels, out_ch=num_classes).to(device)
    
    elif model_name == 'hrnet':
        model = HighResolutionNet(in_ch=in_channels, out_ch=num_classes).to(device)

    elif model_name == 'segnet':
        model = SegNet(in_ch=in_channels, out_ch=num_classes).to(device)
    
    elif model_name == 'fcnresnet':
        model = FCN_ResNet(in_ch=in_channels, out_ch=num_classes).to(device)
    
    elif model_name == 'yournet':
        model = Your_Net(in_ch=in_channels, out_ch=num_classes).to(device)

    

    else:
        raise NotImplementedError(f"model_name {model_name} not supported")


    logger = setup_logger("training_logger", output_file=output_file)
    logger.info("Process started")
    logger.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    
    db_train = DynamicDataset(img_path=imageTr_path, gt_path=labelTr_path, size=args.img_size, augmentations=augmentations)
    db_val = DynamicDataset(img_path=imageVal_path, gt_path=labelVal_path,  size=args.img_size)


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    validloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0


    val_dice_scores = []
    epoch_numbers = []
    for epoch_num in range(max_epoch):
        for i_batch, (img, label, img_name, ori_shape) in enumerate(trainloader):
            image_batch, label_batch = img, label
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            image_batch = image_batch.float()

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
        progress_message = format_progress_message(epoch_num + 1, max_epoch)
        logger.info(progress_message)
           
        save_interval = 5
        if (epoch_num) % save_interval == 0:
            metric_list = []
            model.eval()
            for i_batch, (img, label, img_name, ori_shape) in enumerate(validloader):
                image_batch, label_batch = img, label
                image_batch = image_batch.to(device)
                image_batch = image_batch.float()

                outputs = model(image_batch)



                outputs = torch.nn.functional.interpolate(outputs, size=(ori_shape[-2], ori_shape[-1]), mode='nearest')
                pred = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy().astype(np.uint8)

                label_batch = torch.nn.functional.interpolate(label_batch.unsqueeze(0),
                                                              size=(ori_shape[-2], ori_shape[-1]),
                                                              mode='nearest').squeeze_(0).squeeze_(0).cpu().numpy().astype(np.uint8)

                each_metric = []

                for i in range(1, num_classes):
                    each_metric.append(calculate_metric_percase(pred == i, label_batch == i))

                dice = sum(each_metric) / len(each_metric)
                metric_list.append(dice)

            model.train()
            performance = sum(metric_list) / len(metric_list)

            val_dice_scores.append(performance)
            epoch_numbers.append(epoch_num)

            logger.info('epoch %d : mean_dice : %f' % (epoch_num, performance))

            # Plot validation metrics
            plt.figure(figsize=(10, 10))
            plt.title('Validation Dice Score over Epochs', fontsize=20)
            plt.xlabel('Epochs', fontsize=20)
            plt.ylabel('Dice Score', fontsize=20)
            plt.plot(epoch_numbers, val_dice_scores)
            plt.xticks(epoch_numbers)
            plt.savefig(os.path.join(output_folder, 'progress.png'))


            if performance > best_performance:
                best_performance = performance
                save_mode_path = os.path.join(output_folder, 'checkpoint_final.pth')
                torch.save(model.state_dict(), save_mode_path)
                logger.info("save model to {}".format(save_mode_path))


    logger.info("Training Process finished")




if __name__ == '__main__':
    train(batch_size=args.batch_size, max_epochs=args.max_epochs, base_lr=args.base_lr, seed=args.seed, n_gpu=args.n_gpu, model_name=os.environ['MODEL_NAME'])
    