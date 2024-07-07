import os
from PIL import Image
import numpy as np

folder_path_list = ['E:\\Demo_dataset\\Kvasir-SEG\\train\\image','E:\\Demo_dataset\\Kvasir-SEG\\test\\image']
for folder_path in folder_path_list:
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, file))
            folder_path_new = folder_path+'_png'
            os.makedirs(folder_path_new, exist_ok=True)
            img.save(os.path.join(folder_path_new, file.replace('.jpg', '.png')))
            print(f'{file} converted to png')



folder_path_list = ['E:\\Demo_dataset\\Kvasir-SEG\\train\\label','E:\\Demo_dataset\\Kvasir-SEG\\test\\label']

for folder_path in folder_path_list:
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, file))
            # threshold the image
            img = np.array(img)
            img[img > 100] = 255
            img[img <= 100] = 0
            img = Image.fromarray(img)

            img = img.convert('L')
            folder_path_new = folder_path+'_png'
            os.makedirs(folder_path_new, exist_ok=True)

            img.save(os.path.join(folder_path_new, file.replace('.jpg', '.png')))
            print(f'{file} converted to png')

            