from PIL import Image
import os



# 定义五个文件夹的名称
folders = [ "unet_result", "swin_result", "trans_result", "autosam_result", "nnunet_result", "nnsam_result"]
# 图片的名称，例如 "image.jpg"
label_fold = "GroundTruth"
label_list = os.listdir(label_fold)
out_fold = "draw_result"
os.makedirs(out_fold, exist_ok=True)


dice_all = []
asd_all = []

for fold_first in folders:

    for fold_second in range(1, 6):
        dice_list = [fold_first]
        asd_list = [fold_first]
        img_name = os.path.join(fold_first, str(fold_second), 'metric.txt')
        print(img_name)
        with open(img_name, 'r') as f:
            text = f.read().replace('\n', ' ').replace('[', '').replace(']','')
            metric_list = text.split(' ')
            metric_list = [i for i in metric_list if i != '']
            metric_list = [i for i in metric_list if i != ' ']
            dice_list.append(fold_second*4)
            asd_list.append(fold_second*4)
            for i in range(0, 5):

                dice_list.append(str(round(float(metric_list[i*2])*100,2)) + ' $\pm$ ' + str(round(float(metric_list[i*2+10])*100,2)) )
                asd_list.append(str(round(float(metric_list[i*2+1]),2))   + ' $\pm$ ' + str(round(float(metric_list[i*2+11]),2)) )



        dice_all.append(dice_list)
        asd_all.append(asd_list)
    print(dice_all)
    print(asd_all)

# save into csv file
import pandas as pd
df = pd.DataFrame(dice_all)
df.to_csv('dice.csv', index=False, header=False)
df = pd.DataFrame(asd_all)
df.to_csv('asd.csv', index=False, header=False)
