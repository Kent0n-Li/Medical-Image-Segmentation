import os

file_path = 'E:/nnSAM/nnUNET/nnUNet_raw/Dataset122_BRATS/labelsTr'
file_list = os.listdir(file_path)
for file_name in file_list:
    ori_name = file_name
    new_name = file_name.replace('t1ce.nii.gz', '0000.nii.gz').replace('t1.nii.gz', '0001.nii.gz').replace('t2.nii.gz', '0002.nii.gz').replace('flair.nii.gz', '0003.nii.gz').replace('_seg.nii.gz', '.nii.gz')
    os.rename(os.path.join(file_path, ori_name), os.path.join(file_path, new_name))
    print(ori_name, '======>', new_name)

