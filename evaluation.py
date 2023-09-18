from medpy import metric
import numpy as np
import os
import cv2
import SimpleITK as sitk


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
    


label_path = '/home/s161901/data-yxl/nnSAM/nnUNet_raw/Dataset270_ctmr/labelsTs'
prediction_path = '/home/s161901/data-yxl/nnSAM/project_TransUNet/autosam_result/Dataset284_ctmr2v2'
test_save_path =prediction_path + '_metric'

image_path = '/home/s161901/data-yxl/nnSAM/nnUNet_raw/Dataset270_ctmr/imagesTs'

# Custom colormap for different i values
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Blue, Green, Red, Cyan, Magenta


os.makedirs(test_save_path, exist_ok=True)

label_list = os.listdir(label_path)

# We're going to use a dictionary to gather the files per case.
file_dict = []
metric_final = 0.0

for filename in label_list:

    prediction_name = os.path.join(prediction_path, filename)
    label_name = os.path.join(label_path, filename)
    image_name = os.path.join(image_path, filename.replace('.png','_0000.png'))
    #print(prediction_name)
    pred = cv2.imread(prediction_name, cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)



    prediction = pred
    label = seg

    metric_list = []
    for i in range(1, 6):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
         # Draw the mask on the image using the specified color
        mask = np.where(pred == i, 1, 0).astype(np.uint8)
        colored_mask = np.zeros_like(img)
        colored_mask[mask == 1] = colors[i - 1]
        img = cv2.addWeighted(img, 1, colored_mask, 0.3, 0)

    
    print(metric_list)
    file_dict.append(metric_list)

    metric_final += np.array(metric_list)

    save_path = os.path.join(test_save_path, filename)
    cv2.imwrite(save_path, img)



std = np.std(file_dict, axis=0)
metric_final = np.mean(file_dict, axis=0)
#save in txt
with open(test_save_path + '/metric.txt', 'w') as f:
    f.write(str(metric_final))
    f.write('\n')
    f.write('\n')
    f.write(str(std))

performance = np.mean(np.mean(file_dict, axis=1),axis=0)[0]
mean_hd95 = np.mean(np.mean(file_dict, axis=1),axis=0)[1]

dice_std = np.std(np.mean(file_dict, axis=1),axis=0)[0]
std_hd95 = np.std(np.mean(file_dict, axis=1),axis=0)[1]

print(f"Performance: {performance}")

print(f"Dice std: {dice_std}")
print(f"Mean ASD: {mean_hd95}")
print(f"ASD std: {std_hd95}")


with open(test_save_path + '/performance.txt', 'w') as f:
    f.write(str(performance))
    f.write('\n')
    f.write(str(dice_std))
    f.write('\n')
    f.write(str(mean_hd95))
    f.write('\n')
    f.write(str(std_hd95))
