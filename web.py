from flask import Flask, request, jsonify, render_template, redirect
import subprocess
import threading
from threading import Lock
import os
import psutil
import sys
import glob
import json
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from PIL import Image
import numpy as np
import SimpleITK as sitk
import webbrowser
import cv2
app = Flask(__name__)


colors = [
    (0, 0, 255),    # 红
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 紫
    (255, 255, 0),  # 青
    (192, 0, 0),    # 深红
    (0, 192, 0),    # 深绿
    (0, 0, 192),    # 深蓝
    (192, 192, 0),  # 橄榄
    (192, 0, 192),  # 紫罗兰
    (0, 192, 192),  # 蓝绿
    (128, 128, 128),# 灰
    (128, 0, 0),    # 半深红
    (0, 128, 0),    # 半深绿
    (0, 0, 128),    # 半深蓝
    (128, 128, 0),  # 半深橄榄
    (128, 0, 128),  # 半深紫
    (0, 128, 128),  # 半深蓝绿
    (64, 128, 0),   # 橄榄/绿
    (128, 64, 0),   # 橄榄/红
    (64, 0, 128),   # 紫/蓝
    (128, 0, 64),   # 紫/红
    (0, 128, 64),   # 蓝绿/绿
    (0, 64, 128),   # 蓝绿/蓝
    (64, 0, 0),     # 暗红
    (0, 64, 0),     # 暗绿
    (0, 0, 64),     # 暗蓝
    (64, 64, 64),   # 暗灰
    (64, 64, 0),    # 暗橄榄
    (0, 64, 64),    # 暗蓝绿
    (64, 0, 64)     # 暗紫
]




# 用于保存正在运行的进程
current_process = None
process_lock = Lock()
# 全局变量，用于保存当前命令的输出
current_output = ""

# 用于保存进程的状态 ('running', 'completed', 'not_started')
process_status = 'not_started'
output_file = 'command_output.txt'
conda_env = os.path.basename(sys.prefix)

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print("No such process")
        return

    children = parent.children(recursive=True)

    for child in children:
        child.kill()

    parent.kill()



def calculate_dice(pred, gt):
    # 用于计算 Dice 相似系数
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    
    if union == 0:
        return 1.0  # 如果两者都是空集，Dice 应为 1
    
    return 2 * intersection / union

def calculate_asd(pred, gt):
    # 计算平均表面距离（这里是一个简化的版本，可能需要更高级的算法来计算准确的表面距离）
    pred_border = np.logical_xor(pred, np.roll(pred, 1, axis=0))
    gt_border = np.logical_xor(gt, np.roll(gt, 1, axis=0))

    pred_border_indices = np.argwhere(pred_border)
    gt_border_indices = np.argwhere(gt_border)
    
    if len(pred_border_indices) == 0 or len(gt_border_indices) == 0:
        return 0.0  # 无法计算表面距离
    
    distances = np.linalg.norm(pred_border_indices[:, np.newaxis] - gt_border_indices[np.newaxis, :], axis=-1)
    asd = np.mean(np.min(distances, axis=1)) + np.mean(np.min(distances, axis=0))
    asd /= 2.0
    
    return asd

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_dice(pred, gt)
        asd = calculate_asd(pred, gt)
        return dice, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def reader_thread(process, f):
    for line in iter(process.stdout.readline, ''):
        print(line)
        f.write(line)
    
def run_command_async(command):
    global current_output, process_status, current_process
    try:
        with process_lock:

            if os.path.exists('static/progress.png'):
                white_image = Image.new("RGB", (50, 50), "white")
                white_image.save('static/progress.png')
            if os.path.exists('static/result_visiual.png'):
                white_image = Image.new("RGB", (50, 50), "white")
                white_image.save('static/result_visiual.png')


            current_output = "Output: "
            process_status = 'running'
            env = os.environ.copy()

            current_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

            print("Process started")
            with open(output_file, 'w') as f:
                f.write("Process started")
                f.flush()
                for line in current_process.stdout:
                    print(line)
                    f.write(line)
                    f.flush()


        with process_lock:
            print("Process completed")
            process_status = 'completed'
            current_process = None
            
        
    except Exception as e:
        with process_lock:
            process_status = 'not_started'
            current_process = None


def read_paths_from_file(filename):
    paths = []
    try:
        with open(filename, 'r') as f:
            paths = f.readlines()
        paths = [path.strip() for path in paths]
    except FileNotFoundError:
        print(f"{filename} not found, using default paths.")
    except Exception as e:
        print(f"An error occurred while reading {filename}: {e}")
    return paths

def write_paths_to_file(filename, paths):
    try:
        with open(filename, 'w') as f:
            f.write("\n".join(paths))
    except Exception as e:
        print(f"An error occurred while writing to {filename}: {e}")


def find_latest_txt_file(directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    if not txt_files:
        return None
    
    latest_file = max(txt_files, key=os.path.getmtime)
    
    return latest_file



@app.route('/get_paths', methods=['GET'])
def get_paths():
    paths = read_paths_from_file('paths.txt')

    os.environ['nnUNet_raw'] = os.path.join(paths[0],'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join( paths[0],'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join( paths[0],'nnUNet_results')
    os.environ['MODEL_NAME'] = paths[1]
    os.environ['current_dataset'] = paths[2]
    
    raw_path = os.path.join(paths[0],'nnUNet_raw')
    try:
        all_files_and_folders = os.listdir(raw_path)
        dataset_list = [f for f in all_files_and_folders if os.path.isdir(os.path.join(raw_path, f))]
    except:
        dataset_list = []

    return jsonify({
        'nnUNet_path': paths[0],
        'model_name': paths[1],
        'dataset_list': dataset_list,
        'dataset':paths[2]
    })

    

@app.route('/set_paths_and_model', methods=['POST'])
def set_paths_and_model():
    nnUNet_path = request.json.get('nnUNet_path', '')
    model_name = request.json.get('model_name', '')
    dataset = request.json.get('dataset', '') if request.json.get('dataset', '') else 'Dataset'
    evaluation_type = request.json.get('evaluation_type', '')

    try:
        os.environ['nnUNet_raw'] = os.path.join(nnUNet_path, 'nnUNet_raw')
        os.environ['nnUNet_preprocessed'] = os.path.join(nnUNet_path, 'nnUNet_preprocessed')
        os.environ['nnUNet_results'] = os.path.join(nnUNet_path, 'nnUNet_results')
        os.environ['MODEL_NAME'] = model_name
        os.environ['current_dataset'] = dataset
        os.environ['evaluation_type'] = evaluation_type

        os.makedirs(os.environ['nnUNet_raw'], exist_ok=True)
        os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)
        os.makedirs(os.environ['nnUNet_results'], exist_ok=True)

        paths = [nnUNet_path, model_name, dataset]
        write_paths_to_file('paths.txt', paths)

        return jsonify({'status': 'Configuration updated'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/data_preprocess', methods=['POST'])
def data_preprocess():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})

        dataset_id = request.json.get('dataset', '')

        imageTr_path =  os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTr')
        labelTr_path =  os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr')
        imageTs_path =  os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
        labelTs_path =  os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs')
        num_train = len(os.listdir(imageTr_path))


        if os.environ['MODEL_NAME'] == 'nnunet3d':
            for label_name in os.listdir(labelTr_path):
                label_path = os.path.join(labelTr_path, label_name)
                file_ext = os.path.splitext(label_name)[1]
                
                if file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    nplabel = sitk.ReadImage(label_path)
                    nplabel = sitk.GetArrayFromImage(nplabel)
                else:
                    return jsonify({'status': 'Please use nii.gz, nrrd or mha format.'})
                
                unique_values = np.unique(np.array(nplabel))

                '''nplabel = sitk.ReadImage(label_path)
                nplabel_array = sitk.GetArrayFromImage(nplabel)
                new_label_array = np.searchsorted(unique_values, nplabel_array)
                new_label_sitk = sitk.GetImageFromArray(new_label_array.astype(np.uint8))
                new_label_sitk.CopyInformation(nplabel)
                sitk.WriteImage(new_label_sitk, label_path)

            for label_name in os.listdir(labelTs_path):
                label_path = os.path.join(labelTs_path, label_name)
                file_ext = os.path.splitext(label_name)[1]

                if file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    nplabel = sitk.ReadImage(label_path)
                    nplabel = sitk.GetArrayFromImage(nplabel)
                else:
                    return jsonify({'status': 'Please use nii.gz, nrrd or mha format.'})
                
                unique_values = np.unique(np.array(nplabel))'''



            label_info_set = {}
            label_info_set['background'] = 0
            for i in range(1,len(unique_values)):
                label_info_set['lab'+str(unique_values[i])] = i
            

            #find max
            
            num_list = [int(num.split('_')[-1].split('.')[0]) for num in os.listdir(imageTr_path)]
            max_num = max(num_list)

            image_info_set = {}
            for i in range(max_num+1):
                image_info_set[str(i)] = 'channel'+str(i)

            if file_ext == '.gz':
                file_ext = '.nii.gz'
            num_train = int(num_train/(max_num+1))

            generate_dataset_json(os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset']), image_info_set, label_info_set,
                                num_train, file_ext, dataset_name=os.environ['current_dataset'])

        else:
            if not os.path.exists(os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr_backup')):
                shutil.copytree(labelTr_path, os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr_backup'))
                shutil.copytree(labelTs_path, os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs_backup'))


            label_file_list = []
            count = 0
            for label_name in os.listdir(labelTr_path):
                label_path = os.path.join(labelTr_path, label_name)
                file_ext = os.path.splitext(label_name)[1]
                count += 1
                if count > 20:
                    break

                if file_ext in ['.png', '.bmp', '.tif']:
                    nplabel = Image.open(label_path).convert("L")
                    nplabel = np.array(nplabel)
                    label_file_list.append(nplabel)
                elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    nplabel = sitk.ReadImage(label_path)
                    nplabel = sitk.GetArrayFromImage(nplabel)
                    label_file_list.append(nplabel)
                else:
                    return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})

            unique_values = np.unique(np.array(label_file_list))
            print(unique_values)

            for label_name in os.listdir(labelTr_path):
                label_path = os.path.join(labelTr_path, label_name)
                file_ext = os.path.splitext(label_name)[1]

                if file_ext in ['.png', '.bmp', '.tif']:
                    nplabel = Image.open(label_path).convert("L")
                    nplabel = np.array(nplabel)
                elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    nplabel = sitk.ReadImage(label_path)
                    nplabel = sitk.GetArrayFromImage(nplabel)
                else:
                    return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})
                
                new_label = np.searchsorted(unique_values, nplabel)
                new_label = Image.fromarray(new_label.astype(np.uint8))
                new_label.save(label_path)


            for label_name in os.listdir(labelTs_path):
                label_path = os.path.join(labelTs_path, label_name)
                file_ext = os.path.splitext(label_name)[1]

                if file_ext in ['.png', '.bmp', '.tif']:
                    nplabel = Image.open(label_path).convert("L")
                    nplabel = np.array(nplabel)
                    new_label = np.searchsorted(unique_values, nplabel)
                    new_label = Image.fromarray(new_label.astype(np.uint8), "L")
                    new_label.save(label_path)
                elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    nplabel = sitk.ReadImage(label_path)
                    nplabel_array = sitk.GetArrayFromImage(nplabel)
                    new_label_array = np.searchsorted(unique_values, nplabel_array)
                    new_label_sitk = sitk.GetImageFromArray(new_label_array.astype(np.uint8))
                    new_label_sitk.CopyInformation(nplabel)
                    sitk.WriteImage(new_label_sitk, label_path)
                else:
                    return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})
                


            label_info_set = {}
            label_info_set['background'] = 0
            for i in range(1,len(unique_values)):
                label_info_set['lab'+str(unique_values[i])] = i



            imageTr_list = os.listdir(imageTr_path)

            img_name = imageTr_list[0]
            img_path = os.path.join(imageTr_path, img_name)
            file_ext = os.path.splitext(img_name)[1]
            if file_ext in ['.png', '.bmp', '.tif']:
                npimg = cv2.imread(img_path)
                npimg = np.array(npimg)
            elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                npimg = sitk.ReadImage(img_path)
                npimg = sitk.GetArrayFromImage(npimg)
            else:
                return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})
            
            image_info_set = {}
            if len(npimg.shape) == 2:
                image_info_set['0'] = 'channel0'
            elif len(npimg.shape) == 3:
                for i in range(npimg.shape[2]):
                    image_info_set[str(i)] = 'channel'+str(i)

            if file_ext == '.gz':
                file_ext = '.nii.gz'

            generate_dataset_json(os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset']), image_info_set, label_info_set,
                                num_train, file_ext, dataset_name=os.environ['current_dataset'])


        dataset_id = dataset_id.split('_')[0].replace('Dataset', '')
        
        complete_command = f"conda activate {conda_env} && nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
        print(complete_command)

        threading.Thread(target=run_command_async, args=(complete_command,)).start()
        return jsonify({'status': 'Preprocessing started'})


@app.route('/train_model', methods=['POST'])
def train_model():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})

        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
        if os.environ['MODEL_NAME'] == 'nnunet3d':
            nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

        dataset_id = request.json.get('dataset', '')
        fold = request.json.get('fold', '')
        evaluation_type = request.json.get('evaluation_type', '')

        if evaluation_type == 'test':
            fold = '0'

        os.environ['current_dataset'] = dataset_id
        os.environ['current_fold'] = fold
        
        if os.environ['MODEL_NAME'] == 'nnunet' or os.environ['MODEL_NAME'] == 'nnsam':
            dataset_id = dataset_id.split('_')[0].replace('Dataset', '')
            complete_command = f"conda activate {conda_env} && nnUNetv2_train {dataset_id} 2d {fold}"
        elif os.environ['MODEL_NAME'] == 'nnunet3d':
            dataset_id = dataset_id.split('_')[0].replace('Dataset', '')
            complete_command = f"conda activate {conda_env} && nnUNetv2_train {dataset_id} 3d_fullres {fold}"

        else:
            split_json_path = os.path.join(os.environ['nnUNet_preprocessed'], dataset_id, 'splits_final.json')
            if not os.path.exists(split_json_path):
                complete_command = f"conda activate {conda_env} && nnUNetv2_train {dataset_id} 2d {fold} && python train.py"
            else:
                complete_command = f"conda activate {conda_env} && python train.py"

        print(complete_command)

        threading.Thread(target=run_command_async, args=(complete_command,)).start()
        return jsonify({'status': complete_command})


@app.route('/run_test', methods=['POST'])
def run_test():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})


    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    if os.environ['MODEL_NAME'] == 'nnunet3d':
        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    dataset_id = request.json.get('dataset', '')
    fold = request.json.get('fold', '')
    evaluation_type = request.json.get('evaluation_type', '')

    os.environ['current_dataset'] = dataset_id
    os.environ['current_fold'] = fold


    split_json_path = os.path.join(os.environ['nnUNet_preprocessed'], dataset_id, 'splits_final.json')
    with open(split_json_path, 'r') as f:
        split_json = json.load(f)
    for i, fold_data in enumerate(split_json):
        val_folder = os.path.join(os.environ['nnUNet_preprocessed'], dataset_id, f'fold_{i}_val')
        if not os.path.exists(val_folder):
            os.makedirs(val_folder, exist_ok=True)
            src_dir_path = os.path.join(os.environ['nnUNet_raw'], dataset_id, 'imagesTr')
            src_list = os.listdir(src_dir_path)
            for img_name in fold_data['val']:
                for file in src_list:
                    end_legth = len(file.split('_')[-1])

                    if img_name == file[:-end_legth-1]:
                        src_path = os.path.join(os.environ['nnUNet_raw'], dataset_id, 'imagesTr', file)
                        dest_path = os.path.join(val_folder, file)
                        shutil.copy(src_path, dest_path)
    
    dataset_id = dataset_id.split('_')[0].replace('Dataset', '')

    ckpt_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, f'fold_{fold}','checkpoint_final.pth')
    if not os.path.exists(ckpt_path):
        shutil.copy(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, f'fold_{fold}','checkpoint_best.pth'), ckpt_path)

    if evaluation_type == 'test':
        fold = '0'
        os.environ['current_fold'] = fold
        input_folder = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
        output_folder = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'test_pred')
        os.makedirs(output_folder, exist_ok=True)

        if os.environ['MODEL_NAME'] == 'nnunet' or os.environ['MODEL_NAME'] == 'nnsam':
            complete_command = f"conda activate {conda_env} && nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 2d -f {fold}"
        elif os.environ['MODEL_NAME'] == 'nnunet3d':
            complete_command = f"conda activate {conda_env} && nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 3d_fullres -f {fold}"
        else:
            complete_command = f"conda activate {conda_env} && python test.py"

        
        print(input_folder)

        print(complete_command)
        threading.Thread(target=run_command_async, args=(complete_command,)).start()

    else:
        os.environ['5fold_valid'] = 'True'
        input_folder = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['current_dataset'], f'fold_{fold}_val')
        output_folder = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, f'fold_{fold}', 'validation_pred')
        os.makedirs(output_folder, exist_ok=True)
        if os.environ['MODEL_NAME'] == 'nnunet' or os.environ['MODEL_NAME'] == 'nnsam':
            complete_command = f"conda activate {conda_env} && nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 2d -f {fold}"
        elif os.environ['MODEL_NAME'] == 'nnunet3d':
            complete_command = f"conda activate {conda_env} && nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 3d_fullres -f {fold}"
        else:
            complete_command = f"conda activate {conda_env}"
        print(complete_command)
        threading.Thread(target=run_command_async, args=(complete_command,)).start()


    with process_lock:
        print(input_folder)
        test_img_list_ori = os.listdir(input_folder)
        test_img_list = [test_img for test_img in test_img_list_ori if '_0000.' in test_img]
        metric_list = []
        for test_img_name in test_img_list:
            img_path = os.path.join(input_folder, test_img_name)
            ground_truth_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs', test_img_name.replace('_0000',''))
            prediction_path = os.path.join(output_folder, test_img_name.replace('_0000',''))

            file_ext = os.path.splitext(img_path)[1]

            data_json_file = os.path.join(os.environ['nnUNet_preprocessed'], os.environ['current_dataset'], 'dataset.json')
            with open(data_json_file, 'r') as f:
                data_json = json.load(f)
            label_num = len(data_json['labels'])


            if os.environ['MODEL_NAME'] == 'nnunet3d':
                if file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    pred = sitk.ReadImage(prediction_path)
                    pred = sitk.GetArrayFromImage(pred)
                    ground_truth = sitk.ReadImage(ground_truth_path)
                    ground_truth = sitk.GetArrayFromImage(ground_truth)
                else:
                    return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})


                half_layer = int(pred.shape[0]/2)
                mask_result = np.zeros_like(pred[half_layer])
                mask_result = np.repeat(mask_result[:, :, np.newaxis], 3, axis=2)
                each_metric = []
                for i in range(1, label_num):
                    each_metric.append(calculate_metric_percase(pred == i, ground_truth == i))
                    mask = np.where(pred[half_layer] == i, 1, 0).astype(np.uint8)
                    mask_result[mask == 1] = colors[i - 1]

                metric_list.append(each_metric)

                img_with_mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'visualization_result', test_img_name+'.png')
                os.makedirs(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'visualization_result'), exist_ok=True)
                cv2.imwrite(img_with_mask_save_path, mask_result)
                mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'mask_result', test_img_name+'.png')
                os.makedirs(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'mask_result'), exist_ok=True)
                cv2.imwrite(mask_save_path, mask_result)




            else:

                if file_ext in ['.png', '.bmp', '.tif']:
                    
                    img = Image.open(img_path)
                    img = np.array(img)
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

                    pred = Image.open(prediction_path)
                    pred = np.array(pred)
                    ground_truth = Image.open(ground_truth_path)
                    ground_truth = np.array(ground_truth)


                elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
                    img = sitk.ReadImage(img_path)
                    img = sitk.GetArrayFromImage(img)
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    pred = sitk.ReadImage(prediction_path)
                    pred = sitk.GetArrayFromImage(pred)
                    ground_truth = sitk.ReadImage(ground_truth_path)
                    ground_truth = sitk.GetArrayFromImage(ground_truth)

                else:
                    return jsonify({'status': 'Please use png, bmp, tif, nii.gz, nrrd or mha format.'})
                

                each_metric = []
                mask_result = np.zeros_like(img)
                for i in range(1, label_num):
                    each_metric.append(calculate_metric_percase(pred == i, ground_truth == i))
                    mask = np.where(pred == i, 1, 0).astype(np.uint8)
                    colored_mask = np.zeros_like(img)
                    colored_mask[mask == 1] = colors[i - 1]
                    img = cv2.addWeighted(img, 1, colored_mask, 0.9, 0)
                    mask_result[mask == 1] = colors[i - 1]

                metric_list.append(each_metric)
                img_with_mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'visualization_result', test_img_name)
                os.makedirs(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'visualization_result'), exist_ok=True)
                cv2.imwrite(img_with_mask_save_path, img)

                mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'mask_result', test_img_name)
                os.makedirs(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'mask_result'), exist_ok=True)
                cv2.imwrite(mask_save_path, mask_result)



        dice_mean = np.mean(np.mean(np.array(metric_list), axis=0), axis=0)[0]
        dice_std = np.std(np.mean(np.array(metric_list), axis=0), axis=0)[0]

        asd_mean = np.mean(np.mean(np.array(metric_list), axis=0), axis=0)[1]
        asd_std = np.std(np.mean(np.array(metric_list), axis=0), axis=0)[1]

        #save into csv
        mean_csv_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'test_result_mean.csv')
        with open(mean_csv_path, 'w') as f:
            f.write('dice_mean,dice_std,asd_mean,asd_std\n')
            f.write(f'{dice_mean},{dice_std},{asd_mean},{asd_std}\n')


        csv_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'test_result.csv')
        with open(csv_path, 'w') as f:
            f.write('dice,asd\n')
            for each_metric in metric_list:
                f.write(f'{each_metric[0][0]},{each_metric[0][1]}\n')
        shutil.copy(img_with_mask_save_path,'static/result_visiual.png')
        with open(output_file, 'a') as f:
            f.write(f'\nimg_with_mask_save_path: {img_with_mask_save_path}\n')
            f.write(f'\nmask_save_path: {mask_save_path}\n')
            f.write(f'\nresult_csv_path: {csv_path}')
            f.write(f'\nDICE: {dice_mean} ± {dice_std}\nASD: {asd_mean} ± {asd_std}')


    return jsonify({'status': complete_command})



@app.route('/summary_result', methods=['POST'])
def summary_result():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})

    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    if os.environ['MODEL_NAME'] == 'nnunet3d':
        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    dataset_id = request.json.get('dataset', '')
    fold = request.json.get('fold', '')
    evaluation_type = request.json.get('evaluation_type', '')

    os.environ['current_dataset'] = dataset_id
    os.environ['current_fold'] = fold
    
    input_folder = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
    test_img_list = os.listdir(input_folder)
    method_list_path = os.path.join(os.environ['nnUNet_results'])
    method_list = os.listdir(method_list_path)

    if not os.environ['MODEL_NAME'] == 'nnunet3d':
        for image_case_name in test_img_list:

            images = []
            for method_name in method_list:
                img_with_mask_save_path = os.path.join(os.environ['nnUNet_results'], method_name, os.environ['current_dataset'], nnUNetPlans, 'visualization_result', image_case_name)
                if os.path.exists(img_with_mask_save_path):
                    img = Image.open(img_with_mask_save_path)
                    images.append(img)

            total_width = sum([img.width for img in images])
            max_height = max([img.height for img in images])

            # 创建一个新的空白图片，大小为所有图片宽度的总和，高度为最大的那张图片的高度
            stitched_image = Image.new('RGB', (total_width, max_height))

            # 将每张图片粘贴到新图片上
            x_offset = 0
            for img in images:
                stitched_image.paste(img, (x_offset,0))
                x_offset += img.width

            output_save_path = os.path.join(os.environ['nnUNet_results'], 'summary',os.environ['current_dataset'], 'visualization_result')
            os.makedirs(output_save_path, exist_ok=True)
            stitched_image.save(os.path.join(output_save_path, image_case_name))

        #save image  into static result_visiual.png
        shutil.copy(os.path.join(output_save_path, image_case_name), 'static/result_visiual.png')

    
    dice_all = []
    asd_all = []
    method_exist_all =[]
    for method_name in method_list:

        csv_path = os.path.join(os.environ['nnUNet_results'], method_name, os.environ['current_dataset'], nnUNetPlans, 'test_result_mean.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                dice_mean = float(lines[1].split(',')[0])
                asd_mean = float(lines[1].split(',')[2])
                dice_std = float(lines[1].split(',')[1])
                asd_std = float(lines[1].split(',')[3])
                dice_all.append(str(round(float(dice_mean)*100,2)) + ' ± ' + str(round(float(dice_std)*100,2)))
                asd_all.append(str(round(float(asd_mean),2)) + ' ± ' + str(round(float(asd_std),2)))
                method_exist_all.append(method_name)
            
    #save into csv
    os.makedirs(os.path.join(os.environ['nnUNet_results'], 'summary',os.environ['current_dataset']), exist_ok=True)
    mean_csv_path = os.path.join(os.environ['nnUNet_results'], 'summary',os.environ['current_dataset'], 'test_result_mean.csv')
    with open(mean_csv_path, 'w') as f:
        f.write('method,dice_mean,asd_mean\n')
        for i in range(len(method_exist_all)):
            f.write(f'{method_exist_all[i]},{dice_all[i]},{asd_all[i]}\n')
    #write csv into command_output.txt
    with open(output_file, 'a') as f:
        f.write('\nSummary result:\n')
        f.write(f'{"method":<20}{"dice_mean":<20}{"asd_mean":<20}\n')
        for i in range(len(method_exist_all)):
            f.write(f'{method_exist_all[i]:<20}{dice_all[i]:<20}{asd_all[i]:<20}\n')



    return jsonify({'status': "Summary result completed"})



@app.route('/edit_network', methods=['GET'])
def edit_network():
    os.system("notepad.exe ./networks/YourNet.py")



@app.route('/get_output', methods=['GET'])
def get_output():
    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    if os.environ['MODEL_NAME'] == 'nnunet3d':
        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    dir_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], nnUNetPlans, 'fold_' + os.environ['current_fold'])

    if os.environ['MODEL_NAME'] == 'nnunet3d':
        dir_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'], 'nnUNetTrainer__nnUNetPlans__3d_fullres', 'fold_' + os.environ['current_fold'])

    try:
        if os.path.exists(dir_path):
        #output_file = find_latest_txt_file(dir_path)
            progress_png = os.path.join(dir_path, 'progress.png')
            try:
                shutil.copy(progress_png, os.path.join('static', 'progress.png'))
            except:
                pass
        with open(output_file, "r") as f:
            lines = f.readlines()
            lines_to_read = lines if len(lines) < 50 else lines[-50:]
            content = " ".join(lines_to_read)
        content = content
        return jsonify({'content': content})

    except:
        return jsonify({'content': 'No output yet'})


@app.route('/get_status', methods=['GET'])
def get_status():
    global process_status
    return jsonify({'status': process_status})


@app.route('/run_command', methods=['POST'])
def run_command():
    global process_status
    with process_lock:
        if process_status == 'running':
            current_output = "Output: "
            for line in current_process.stdout:
                current_output += line
            print(current_output)
            return jsonify({'error': 'A command is already running'})

        command = request.json.get('command', '')
        
        complete_command = f"conda activate {conda_env} && {command}"
        print(complete_command)

        threading.Thread(target=run_command_async, args=(complete_command,)).start()
        return jsonify({'status': complete_command})



@app.route('/stop_command', methods=['POST'])
def stop_command():
    global current_process
    if current_process:
        try:
            print("Terminating process...")
            kill_process_tree(current_process.pid)
            current_process = None
            return jsonify({'status': 'Command stopped'})
        except:
            return jsonify({'error': 'Stopped'})
    else:
        return jsonify({'error': 'No command is running'})




@app.route('/')
def index():
    return render_template("index.html")  # 渲染 index.html


if __name__ == '__main__':
    if not os.path.exists('paths.txt'):
        write_paths_to_file('paths.txt', [os.getcwd(),"nnunet","Dataset011_example"])

    if os.environ.get('current_dataset') is None:
        paths = read_paths_from_file('paths.txt')
        os.environ['nnUNet_raw'] = os.path.join(paths[0],'nnUNet_raw')
        os.environ['nnUNet_preprocessed'] = os.path.join( paths[0],'nnUNet_preprocessed')
        os.environ['nnUNet_results'] = os.path.join( paths[0],'nnUNet_results')
        os.environ['MODEL_NAME'] = paths[1]
        os.environ['current_dataset'] = paths[2]

    with open(output_file, 'w') as f:
            f.write("\n")
    if os.environ.get('current_fold') is None:
        os.environ['current_fold'] = '0'

    webbrowser.open("http://127.0.0.1:5000/", new=2)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
