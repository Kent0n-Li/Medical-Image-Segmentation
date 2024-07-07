import random
from multiprocessing  import Lock
from flask import Flask, request, jsonify, render_template
from multiprocessing import Process
import psutil
import glob
import json
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from PIL import Image
import numpy as np
import SimpleITK as sitk
import webbrowser
import cv2
from config import *
from scipy.spatial import cKDTree
import time
from test import test_model
from train import train
import subprocess
import threading
app = Flask(__name__)

# 用于保存正在运行的进程
current_process = None
process_lock = Lock()
# 全局变量，用于保存当前命令的输出
current_output = ""
# 用于保存进程的状态 ('running', 'completed', 'not_started')
os.environ['process_status'] = 'not_started'

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
    pred_border = np.logical_xor(pred, np.roll(pred, 1, axis=0))
    gt_border = np.logical_xor(gt, np.roll(gt, 1, axis=0))

    pred_border_indices = np.argwhere(pred_border)
    gt_border_indices = np.argwhere(gt_border)

    if len(pred_border_indices) == 0 or len(gt_border_indices) == 0:
        return 0.0  # 无法计算表面距离

    tree_gt = cKDTree(gt_border_indices)
    tree_pred = cKDTree(pred_border_indices)

    distances_to_gt = tree_gt.query(pred_border_indices)[0]
    distances_to_pred = tree_pred.query(gt_border_indices)[0]

    asd = np.mean(distances_to_gt) + np.mean(distances_to_pred)
    asd /= 2.0

    return asd


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_dice(pred, gt)
        asd = calculate_asd(pred, gt)
        return [dice, asd]
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def reader_thread(process, f):
    for line in iter(process.stdout.readline, ''):
        print(line)
        f.write(line)




def run_command_async(command):
    global current_output, current_process

    print("Starting run_command_async")
    try:
        with process_lock:
            print("Starting run_command_async")
            if os.path.exists('static/progress.png'):
                white_image = Image.new("RGB", (50, 50), "white")
                white_image.save('static/progress.png')
            if os.path.exists('static/result_visiual.png'):
                white_image = Image.new("RGB", (50, 50), "white")
                white_image.save('static/result_visiual.png')

        
            current_output = "Output: "
            os.environ['process_status'] = 'running'
            env = os.environ.copy()
            print(command)
            current_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, env=env)

            print("Process started")
            with open(output_file, 'w') as f:
                f.write("Process started")
                f.flush()
                for line in current_process.stdout:
                    print(line)
            #        f.write(line)
            #        f.flush()

        with process_lock:
            print("Process completed")
            os.environ['process_status'] = 'completed'
            print("set Process completed")
            current_process = None


    except Exception as e:
        with process_lock:
            os.environ['process_status'] = 'not_started'
            current_process = None


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
    


def write_output_from_queue_to_file(output_queue):
    with open(output_file, 'w') as f:
        while True:
            message = output_queue.get()  # This will block until a message is available
            if message == "Process completed":  # Assuming this is your termination condition
                print(message)
                f.write(message + '\n')
                f.flush()
                break  # Exit the loop
            else:
                print(message)  # Optional: for logging to console as well
                f.write(message + '\n')
                f.flush()


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


def dataset_name_exists(target_name):
    base_path = os.environ['medseg_raw']
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)):
            if item.split('_')[-1] == target_name:
                return True
    return False


def resize_image(mask_path, output_path, new_size):
    file_ext = os.path.splitext(mask_path)[1]
    if file_ext in ['.png', '.bmp', '.tif', '.jpg', '.jpeg']:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_resized = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, mask_resized)
        print_web(f"Image resized: {output_path}")


@app.route('/full_auto', methods=['POST'])
def full_auto():
    data = request.json
    selected_models = data.get('models_list', [])
    dataset = data.get('dataset')
    batchSize = data.get('batchSize', '4')
    totalEpochs = data.get('totalEpochs', '100')
    learningRate = data.get('learningRate', '0.01')

    try:
        for model in selected_models:
            with app.test_client() as client:
                # Call train_model endpoint
                response = client.post('/train_model', json={
                    'model_name': model,
                    'dataset': dataset,
                    'batchSize': batchSize,
                    'totalEpochs': totalEpochs,
                    'learningRate': learningRate
                })
                print(response.json)

                response = client.post('/run_test', json={
                    'model_name': model,
                    'dataset': dataset
                })
                print(response.json)

                response = client.post('/summary_result', json={
                    'model_name': model,
                    'dataset': dataset
                })
                print(response.json)

        return jsonify({"status": "Full auto completed successfully."})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})


@app.route('/get_paths', methods=['GET'])
def get_paths():
    raw_path = os.environ['medseg_raw']
    try:
        all_files_and_folders = os.listdir(raw_path)
        dataset_list = [f for f in all_files_and_folders if os.path.isdir(os.path.join(raw_path, f))]
    except:
        dataset_list = []

    return jsonify({
        'model_name': os.environ['MODEL_NAME'],
        'dataset_list': dataset_list,
        'dataset': os.environ['current_dataset'],
    })


@app.route('/import_dataset', methods=['POST'])
def import_dataset():
    training_image_path = request.json.get('training_image_path', '')
    training_label_path = request.json.get('training_label_path', '')
    testing_image_path = request.json.get('testing_image_path', '')
    testing_label_path = request.json.get('testing_label_path', '')
    dataset_name = request.json.get('dataset_name', '')
    image_size = int(request.json.get('imageSize', '256'))

    if dataset_name_exists(dataset_name):
        return jsonify({'error': f"Dataset with name {dataset_name} already exists!"})


    dataset_id = len(os.listdir(os.environ['medseg_raw'])) + 1
    dataset_id = "{:03}".format(dataset_id)
    dataset_name = f"Dataset{dataset_id}_{dataset_name}"
    os.environ['current_dataset'] = dataset_name

    target_paths = {
        'training_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTr'),
        'training_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTr'),
        'testing_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs'),
        'testing_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs'),
        'validation_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesVal'),
        'validation_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsVal'),
    }

    try:
        # create target directories
        for path_each in target_paths.values():
            os.makedirs(path_each, exist_ok=True)

        train_img_list = os.listdir(training_image_path)
        split_index = random_shuffle_with_spilt_index(train_img_list, validation_factor)

        # Validation dataset
        val_img_list = train_img_list[:split_index]
        train_img_list = train_img_list[split_index:]

        # Copy and resize images for training, validation, and testing datasets
        copy_and_resize_images(training_image_path, target_paths['training_image'], train_img_list, image_size)
        copy_and_resize_images(training_label_path, target_paths['training_label'], train_img_list, image_size)
        copy_and_resize_images(training_image_path, target_paths['validation_image'], val_img_list, image_size)
        copy_and_resize_images(training_label_path, target_paths['validation_label'], val_img_list, image_size)

        # For testing dataset, no need to split, just copy and resize
        test_img_list = os.listdir(testing_image_path)
        copy_and_resize_images(testing_image_path, target_paths['testing_image'], test_img_list, image_size)
        test_label_list = os.listdir(testing_label_path)
        copy_and_resize_images(testing_label_path, target_paths['testing_label'], test_label_list, image_size)

        with process_lock:
            complete_command = f"conda activate {conda_env} && python data_process.py"
            print_web(complete_command)
            threading.Thread(target=run_command_async, args=(complete_command,)).start()


        return jsonify({'status': 'The dataset is being preprocessed, which may take some time.'})

    except Exception as e:
        return jsonify({'error': str(e)})

def save_dataset_json(dataset_id, dataset_name, image_size, img_channel, label_class_num):
    dataset_json = {
        "id": dataset_id,
        "name": dataset_name,
        "imgae_size": image_size,
        "img_channel": img_channel,
        "label_class_num": label_class_num,
        "RandomBrightnessContrast": "False",
        "brightness_limit_min": "0",
        "brightness_limit_max": "0",
        "contrast_limit_min": "0",
        "contrast_limit_max": "0",
        "RandomRotate90": "False",
        "VerticalFlip": "False",
        "HorizontalFlip": "False",
    }
    with open(os.path.join(os.environ['medseg_raw'], dataset_name, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f)




@app.route('/data_aug', methods=['POST'])
def data_augmentation():
    dataset_id = request.json.get('dataset', '')
    os.environ['current_dataset'] = dataset_id

    #        body: JSON.stringify({ Blur:Blur, blur_limit_min_Blur:blur_limit_min_Blur, blur_limit_max_Blur:blur_limit_max_Blur, RandomBrightnessContrast:RandomBrightnessContrast, brightness_limit_min:brightness_limit_min, brightness_limit_max:brightness_limit_max, contrast_limit_min:contrast_limit_min, contrast_limit_max:contrast_limit_max, RandomRotate90:RandomRotate90, VerticalFlip:VerticalFlip, HorizontalFlip:HorizontalFlip, dataset:dataset })


    RandomBrightnessContrast = request.json.get('RandomBrightnessContrast', 'False')
    brightness_limit_min = request.json.get('brightness_limit_min', '0')
    brightness_limit_max = request.json.get('brightness_limit_max', '0')
    contrast_limit_min = request.json.get('contrast_limit_min', '0')
    contrast_limit_max = request.json.get('contrast_limit_max', '0')
    RandomRotate90 = request.json.get('RandomRotate90', 'False')
    VerticalFlip = request.json.get('VerticalFlip', 'False')
    HorizontalFlip = request.json.get('HorizontalFlip', 'False')

    #add into dataset.json
    data_json_file = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'dataset.json')
    with open(data_json_file) as f:
        json_data = json.load(f)
        json_data['RandomBrightnessContrast'] = RandomBrightnessContrast
        json_data['brightness_limit_min'] = brightness_limit_min
        json_data['brightness_limit_max'] = brightness_limit_max
        json_data['contrast_limit_min'] = contrast_limit_min
        json_data['contrast_limit_max'] = contrast_limit_max
        json_data['RandomRotate90'] = RandomRotate90
        json_data['VerticalFlip'] = VerticalFlip
        json_data['HorizontalFlip'] = HorizontalFlip

    with open(data_json_file, 'w') as f:
        json.dump(json_data, f)
        print(json_data)

        print_web(f"Data augmentation started for {dataset_id}")
    
    return jsonify({'status': 'Data augmentation completed'})


def convert_jpg_to_png_all_from_path(dir_path):
    """Converts all JPG files to PNG format."""
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        if os.path.splitext(img_name)[1] in ['.jpg', '.jpeg']:
            new_path = os.path.join(dir_path, os.path.splitext(img_name)[0] + '.png')
            convert_jpg_to_png(img_path, new_path)


def random_shuffle_with_spilt_index(train_img_list, factor):
    random.shuffle(train_img_list)  # Shuffle the list to ensure random selection
    return int(len(train_img_list) * factor)  # % for validation


def copy_and_resize_images(source_path, target_path, file_list, image_size):
    """
    Copies and resizes images from source to target path.
    """
    for item in file_list:
        source_file = os.path.join(source_path, item)
        target_file = os.path.join(target_path, item)
        shutil.copy2(source_file, target_file)
        print_web(f"Image copied: {target_file}")
        resize_image(target_file, target_file, image_size)


def find_unique_labels(labelTr_path):
    """
    Finds unique labels in the label image.
    """
    label_file_list = []
    count = 0
    for label_name in os.listdir(labelTr_path):
        label_path = os.path.join(labelTr_path, label_name)
        file_ext = os.path.splitext(label_name)[1]
        count += 1
        if count > 30:
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
            return jsonify({'status': 'Please use png, nii.gz, or nii format.'})

    flattened_labels = [np.ravel(label) for label in label_file_list]
    unique_values = np.unique(np.concatenate(flattened_labels))
    print(unique_values)
    print_web(f"Unique values found: {unique_values}")

    return unique_values

def convert_label_by_searchsorted(dir_path, unique_values):
    """
    Converts label image to unique values using searchsorted.
    """
    for label_name in os.listdir(dir_path):
        label_path = os.path.join(dir_path, label_name)
        file_ext = os.path.splitext(label_path)[1]
        if file_ext in ['.png', '.bmp', '.tif']:
            nplabel = Image.open(label_path).convert("L")
            nplabel = np.array(nplabel)
        elif file_ext in ['.gz', '.nrrd', '.mha', '.nii']:
            nplabel = sitk.ReadImage(label_path)
            nplabel = sitk.GetArrayFromImage(nplabel)
        else:
            return jsonify({'status': 'Please use png, nii.gz, or nii format.'})

        new_label = np.searchsorted(unique_values, nplabel)
        new_label = Image.fromarray(new_label.astype(np.uint8))
        new_label.save(label_path)

        print_web(f"Label converted: {label_path}")



def convert_jpg_to_png(jpg_path, png_path):
    """Converts a JPG file to PNG format."""
    with Image.open(jpg_path) as img:
        img.save(png_path, "PNG")
    os.remove(jpg_path)  # Delete the original JPG file after conversion
    print_web(f"JPG to PNG conversion completed for {jpg_path}")


@app.route('/train_model', methods=['POST'])
def train_model():
    print("Starting train_model")
    with process_lock:
        if os.environ['process_status'] == 'running':
            return jsonify({'error': 'A command is already running'})
        model_name = request.json.get('model_name', '')
        os.environ['MODEL_NAME'] = model_name
        dataset_id = request.json.get('dataset', '')
        os.environ['current_dataset'] = dataset_id

        batchSize = request.json.get('batchSize', '4')
        totalEpochs = request.json.get('totalEpochs', '100')
        learningRate = request.json.get('learningRate', '0.01')

        complete_command = f"conda activate {conda_env} && python train.py --batch_size {batchSize} --max_epochs {totalEpochs} --base_lr {learningRate}"

        print(complete_command)

        threading.Thread(target=run_command_async, args=(complete_command,)).start()
        return jsonify({'status': complete_command})


@app.route('/run_test', methods=['POST'])
def run_test():
    
    with process_lock:
        if os.environ['process_status'] == 'running':
            return jsonify({'error': 'A command is already running'})

        model_name = request.json.get('model_name', '')
        os.environ['MODEL_NAME'] = model_name
        dataset_id = request.json.get('dataset', '')
        os.environ['current_dataset'] = dataset_id

        
        
        input_folder = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs')
        output_folder = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'test_pred')
        os.makedirs(output_folder, exist_ok=True)

        complete_command = f"conda activate {conda_env} && python test.py"

        print(input_folder)
        print(complete_command)

        command_thread = threading.Thread(target=run_command_async, args=(complete_command,))
        command_thread.start()


    with process_lock:
    
        complete_command = f"conda activate {conda_env} && python test_save.py"
        print(complete_command)
        threading.Thread(target=run_command_async, args=(complete_command,)).start()
        

    return jsonify({'status': complete_command})


@app.route('/summary_result', methods=['POST'])
def summary_result():
    
    with process_lock:
        if os.environ['process_status'] == 'running':
            return jsonify({'error': 'A command is already running'})

    model_name = request.json.get('model_name', '')
    os.environ['MODEL_NAME'] = model_name
    dataset_id = request.json.get('dataset', '')
    os.environ['current_dataset'] = dataset_id

    input_folder = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs')
    test_img_list = os.listdir(input_folder)
    method_list_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'])
    method_list = os.listdir(method_list_path)

    for image_case_name in test_img_list:
        images = []
        for method_name in method_list:
            img_with_GT_save_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], 'Ground_Truth', 'visualization_result', image_case_name)
            if os.path.exists(img_with_GT_save_path) and len(images) == 0:
                img = Image.open(img_with_GT_save_path)
                images.append(img)

            img_with_mask_save_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], method_name, 'visualization_result', image_case_name)
            if os.path.exists(img_with_mask_save_path):
                img = Image.open(img_with_mask_save_path)
                images.append(img)

        total_width = sum([img.width for img in images])
        max_height = max([img.height for img in images])

        # create a new image with the same height and the sum of the width of the images
        stitched_image = Image.new('RGB', (total_width, max_height))

        # paste the images one after the other
        x_offset = 0
        for img in images:
            stitched_image.paste(img, (x_offset, 0))
            x_offset += img.width

        output_save_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], 'summary_result')
        os.makedirs(output_save_path, exist_ok=True)
        stitched_image.save(os.path.join(output_save_path, image_case_name))

    # save image  into static result_visiual.png
    shutil.copy(os.path.join(output_save_path, image_case_name), 'static/result_visiual.png')

    dice_all = []
    asd_all = []
    method_exist_all = []
    for method_name in method_list:

        csv_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], method_name, 'test_result_mean.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                dice_mean = float(lines[1].split(',')[0])
                asd_mean = float(lines[1].split(',')[2])
                dice_std = float(lines[1].split(',')[1])
                asd_std = float(lines[1].split(',')[3])
                dice_all.append(str(round(float(dice_mean) * 100, 2)) + ' ± ' + str(round(float(dice_std) * 100, 2)))
                asd_all.append(str(round(float(asd_mean), 2)) + ' ± ' + str(round(float(asd_std), 2)))
                method_exist_all.append(method_name)

    # save into csv
    mean_csv_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], 'summary_result', 'summary_result.csv')
    with open(mean_csv_path, 'w') as f:
        f.write('method,dice_mean,asd_mean\n')
        for i in range(len(method_exist_all)):
            f.write(f'{method_exist_all[i]},{dice_all[i]},{asd_all[i]}\n')
    # write csv into command_output.txt
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
    

    dir_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'])

    try:
        if os.path.exists(dir_path):
            # output_file = find_latest_txt_file(dir_path)
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
    
    return jsonify({'status': os.environ['process_status']})





@app.route('/')
def index():
    return render_template("index.html")  # render a template


if __name__ == '__main__':

    os.makedirs(os.path.join(os.getcwd(), 'medseg_raw'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'medseg_results'), exist_ok=True)

    os.environ['medseg_raw'] = os.path.join(os.getcwd(), 'medseg_raw')
    os.environ['medseg_results'] = os.path.join(os.getcwd(), 'medseg_results')

    if os.environ.get('current_dataset') is None:
        os.environ['current_dataset'] = os.listdir(os.environ['medseg_raw'])[0]
        os.environ['MODEL_NAME'] = 'unet'


    with open(output_file, 'w') as f:
        f.write("\n")

    webbrowser.open("http://127.0.0.1:5000/", new=1)
    app.run(debug=True, host='0.0.0.0', port=5000)


