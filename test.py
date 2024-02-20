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
from  config import parse_args
import gdown
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser()
#
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
# parser.add_argument('--max_epochs', type=int,
#                     default=200, help='maximum epoch number to train')
# parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# parser.add_argument('--deterministic', type=int, default=1,
#                     help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float, default=0.01,
#                     help='segmentation network learning rate')
# parser.add_argument('--img_size', type=int,
#                     default=224, help='input patch size of network input')
# parser.add_argument('--seed', type=int,
#                     default=1234, help='random seed')
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                          'full: cache all data, '
#                          'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')

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
    def __init__(self, img_path, gt_path, data_end_json, size=None):

        with open(data_end_json) as f:
            self.file_end = json.load(f)['file_ending']

        self.img_name = os.listdir(img_path)
        self.size = size
        self.img_path = img_path
        self.gt_path = gt_path

    def __getitem__(self, item):
        imagename = self.img_name[item]
        img_path = os.path.join(self.img_path, imagename)

        if self.file_end in ['.png', '.bmp', '.tif']:
            npimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            npimg = np.array(npimg)

        elif self.file_end in ['.gz', '.nrrd', '.mha', '.nii.gz', '.nii']:
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

        return npimg, imagename.replace('_0000', ''), ori_shape

    def __len__(self):
        size = int(len(self.img_name))
        return size


# if __name__ == "__main__":

def test_model():
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # bytes to GB
        args.batch_size = int(total_memory / 10) * 2
    else:
        args.batch_size = 2

    fold = os.environ['current_fold']

    data_json_file = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'dataset.json')
    # split_json_path = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['current_dataset'], 'splits_final.json')
    # base_json_path = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['current_dataset'])
    output_folder_test = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                      os.environ['current_dataset'], 'nnUNetTrainer__nnUNetPlans__2d', 'test_pred')
    output_folder_5fold = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                       os.environ['current_dataset'], 'nnUNetTrainer__nnUNetPlans__2d', f'fold_{fold}')

    os.makedirs(output_folder_test, exist_ok=True)
    os.makedirs(output_folder_5fold, exist_ok=True)

    imageTr_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTr')
    labelTr_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr')
    imageTs_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
    labelTs_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs')

    with open(data_json_file) as f:
        json_data = json.load(f)
        num_classes = len(json_data['labels'])
        in_channels = len(json_data['channel_names'])

    weights_path = os.path.join(output_folder_5fold, 'checkpoint_final.pth')

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

    else:
        raise NotImplementedError(f"model_name {model_name} not supported")

    model.load_state_dict(torch.load(weights_path))

    logging.basicConfig(filename="logging.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    if model_name == 'swinunet' or model_name == 'transunet':
        db_test = DynamicDataset(img_path=imageTs_path, gt_path=labelTs_path, data_end_json=data_json_file,
                                 size=args.img_size)
    else:
        db_test = DynamicDataset(img_path=imageTs_path, gt_path=labelTs_path, data_end_json=data_json_file)

    with open(data_json_file) as f:
        file_end = json.load(f)['file_ending']

    testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    best_performance = 0.0
    val_dice_scores = []
    epoch_numbers = []
    for i_batch, (img, img_name, ori_shape) in enumerate(testloader):
        image_batch = img
        image_batch = image_batch.cuda()
        image_batch = image_batch.float()

        outputs = model(image_batch)

        outputs = torch.nn.functional.interpolate(outputs, size=(ori_shape[-2], ori_shape[-1]), mode='bilinear',
                                                  align_corners=True)
        pred = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        pred = pred.astype(np.uint8)

        if file_end in ['.png', '.bmp', '.tif']:
            pred_img = Image.fromarray(pred)
            pred_img.save(os.path.join(output_folder_test, img_name[0]))

        elif file_end in ['.gz', '.nrrd', '.mha', '.nii.gz', '.nii']:
            pred_img = sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred_img, os.path.join(output_folder_test, img_name[0]))