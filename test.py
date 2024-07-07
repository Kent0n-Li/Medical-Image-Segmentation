import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.UNet import U_Net, R2U_Net, AttU_Net, NestedUNet
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
import sys
import torch.nn as nn
import torch.optim as optim
from networks.swin_config import get_swin_config
import requests
import gdown
import matplotlib.pyplot as plt
from logHelper import setup_logger
from config import output_file, parse_args
from networks.YourNet import Your_Net
from networks.GT_UNet import GT_U_Net
from networks.model.BiSeNet import BiSeNet
from networks.model.DDRNet import DDRNet
from networks.model.DeeplabV3Plus import Deeplabv3plus_res50
from networks.model.FCN_ResNet import FCN_ResNet
from networks.model.HRNet import HighResolutionNet
from networks.SegNet import SegNet

args = parse_args()


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
    def __init__(self, img_path, gt_path, size=None):

        self.img_name = os.listdir(img_path)
        self.size = size
        self.img_path = img_path
        self.gt_path = gt_path

    def __getitem__(self, item):
        imagename = self.img_name[item]
        img_path = os.path.join(self.img_path, imagename)

        file_end = imagename.split('.')[-1]

        if file_end in ['png']:
            npimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            npimg = np.array(npimg)

        elif file_end in ['gz', 'nrrd', 'mha', 'nii.gz', 'nii']:
            npimg = sitk.ReadImage(img_path)
            npimg = sitk.GetArrayFromImage(npimg)

        if npimg.ndim == 2:
            npimg = np.expand_dims(npimg, axis=0)
        elif npimg.ndim == 3:
            npimg = npimg.transpose((2, 0, 1))

        ori_shape = npimg.shape
        npimg = torch.from_numpy(npimg)

        if self.size is not None:
            resize = transforms.Resize(size=(self.size, self.size), antialias=None)
            npimg = resize(npimg)
        else:
            adapt_size = transforms.Resize(size=(int(npimg.shape[1] / 64 + 1) * 64, int(npimg.shape[2] / 64 + 1) * 64),
                                           antialias=None)
            npimg = adapt_size(npimg)

        return npimg, imagename, ori_shape

    def __len__(self):
        size = int(len(self.img_name))
        return size




def test_model():
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_json_file = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'dataset.json')

    with open(data_json_file) as f:
        json_data = json.load(f)
        num_classes = json_data['label_class_num']
        in_channels = json_data['img_channel']
        args.img_size = json_data['imgae_size']


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    args.batch_size = 1


    output_folder_test = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'test_pred')

    os.makedirs(output_folder_test, exist_ok=True)
    


    imageTs_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs')
    labelTs_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs')
    weights_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'checkpoint_final.pth') 


    model_name = os.environ['MODEL_NAME']
    if model_name == 'unet':
        model = U_Net(in_ch=in_channels, out_ch=num_classes).cuda()

    elif model_name == 'transunet':
        download_url = 'https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz'
        vit_name = 'R50-ViT-B_16'
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_classes
        args.img_size = 224
        args.vit_patches_size = 16
        config_vit.n_skip = 3
        config_vit.pretrained_path = './networks/R50+ViT-B_16.npz'
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=224, num_classes=num_classes).cuda()

        model.load_from(weights=np.load('networks/R50+ViT-B_16.npz'))


    elif model_name == 'swinunet':
        args.cfg = './networks/swin_tiny_patch4_window7_224_lite.yaml'
        args.opts = None
        args.img_size = 224
        swin_config = get_swin_config(args)
        model = SwinUnet(swin_config, img_size=224, num_classes=num_classes).cuda()
        url = "https://drive.google.com/uc?id=1TyMf0_uvaxyacMmVzRfqvLLAWSOE2bJR"
        output = swin_config.MODEL.PRETRAIN_CKPT
        model.load_from(swin_config)

    elif model_name == 'unetpp':
        model = NestedUNet(in_ch=in_channels, out_ch=num_classes).cuda()

    elif model_name == 'attunet':
        model = AttU_Net(in_ch=in_channels, out_ch=num_classes).cuda()

    elif model_name == 'r2unet':
        model = R2U_Net(in_ch=in_channels, out_ch=num_classes).cuda()

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

    model.load_state_dict(torch.load(weights_path))

    logger = setup_logger("training_logger", output_file=output_file)
    logger.info("Process started")
    logger.info(str(args))
    base_lr = args.base_lr
    
    db_test = DynamicDataset(img_path=imageTs_path, gt_path=labelTs_path, size=args.img_size)

        

    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.eval()


    for i_batch, (img, img_name, ori_shape) in enumerate(testloader):
        image_batch = img
        image_batch = image_batch.cuda()
        image_batch = image_batch.float()

        outputs = model(image_batch)

        outputs = torch.nn.functional.interpolate(outputs, size=(ori_shape[-2], ori_shape[-1]), mode='bilinear',
                                                  align_corners=True)
        pred = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        pred = pred.astype(np.uint8)
        print(f"Processing {img_name[0]}")

        file_end = img_name[0].split('.')[-1]
        if file_end in ['png', 'bmp', 'tif']:
            pred_img = Image.fromarray(pred)
            pred_img.save(os.path.join(output_folder_test, img_name[0]))

        elif file_end in ['gz', 'nrrd', 'mha', 'nii.gz', 'nii']:
            pred_img = sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred_img, os.path.join(output_folder_test, img_name[0]))

if __name__ == "__main__":
    test_model()
