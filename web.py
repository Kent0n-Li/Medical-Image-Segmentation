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

from test import test_model
from train import train

app = Flask(__name__)

# 用于保存正在运行的进程
current_process = None
process_lock = Lock()
# 全局变量，用于保存当前命令的输出
current_output = ""

# 用于保存进程的状态 ('running', 'completed', 'not_started')
process_status = 'not_started'
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


def run_command_async(command, batch_size=None, max_epochs=None, base_lr=None):
    global current_output, process_status, current_process
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
            process_status = 'running'
            # env = os.environ.copy()
            if command.startswith('train'):
                train(batch_size, max_epochs, base_lr)
            elif command.startswith('test'):
                test_model()

            # print("Process started")
            # with open(output_file, 'w') as f:
            #     f.write("Process started")
            #     f.flush()
            #     for line in current_process.stdout:
            #         print(line)
            #         f.write(line)
            #         f.flush()

        with process_lock:
            print("Process completed")
            process_status = 'completed'
            current_process = None

    except Exception as e:
        print(f"An error occurred: {e}")
        with process_lock:
            process_status = 'not_started'
            current_process = None


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
    base_path = os.environ['nnUNet_raw']
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)):
            if item.split('_')[-1] == target_name:
                return True
    return False


def resize_image(mask_path, output_path, new_size):
    print(mask_path)
    file_ext = os.path.splitext(mask_path)[1]
    if file_ext in ['.png', '.bmp', '.tif']:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, mask_resized)


@app.route('/full_auto', methods=['POST'])
def full_auto():
    data = request.json
    selected_models = data.get('models_list', [])
    dataset = data.get('dataset')

    batchSize = data.get('batchSize', '4')
    totalEpochs = data.get('totalEpochs', '100')
    learningRate = data.get('learningRate', '0.01')

    try:
        response = data_preprocess()

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
    raw_path = os.environ['nnUNet_raw']
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

    # 目标路径（你可以根据需要更改）
    dataset_id = len(os.listdir(os.environ['nnUNet_raw'])) + 1
    dataset_id = "{:03}".format(dataset_id)
    dataset_name = f"Dataset{dataset_id}_{dataset_name}"
    os.environ['current_dataset'] = dataset_name

    # Define target paths
    target_paths = {
        'training_image': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTr'),
        'training_label': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr'),
        'testing_image': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs'),
        'testing_label': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs'),
        'validation_image': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesVal'),
        'validation_label': os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsVal'),
    }
    try:
        # 创建目标文件夹
        for path in target_paths.values():
            os.makedirs(path, exist_ok=True)

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

        return jsonify({'status': 'Dataset imported successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})


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
        resize_image(target_file, target_file, image_size)


@app.route('/data_preprocess', methods=['POST'])
def data_preprocess():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})

        model_name = request.json.get('model_name', '')
        dataset = request.json.get('dataset', '') if request.json.get('dataset', '') else 'Dataset'
        os.environ['MODEL_NAME'] = model_name
        os.environ['current_dataset'] = dataset

        imageTr_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTr')
        labelTr_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTr')
        imageTs_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
        labelTs_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs')
        num_train = len(os.listdir(imageTr_path))

        label_file_list = []
        count = 0
        for label_name in os.listdir(labelTr_path):
            label_path = os.path.join(labelTr_path, label_name)
            file_ext = os.path.splitext(label_name)[1]
            count += 1
            if count > 20:
                break

            # Question: should I use a new loop to convert all the jpg to pngs.
            # Check and convert JPG to PNG before further processing
            if file_ext == '.jpg':
                png_label_name = os.path.splitext(label_name)[0] + '.png'
                png_label_path = os.path.join(labelTr_path, png_label_name)
                convert_jpg_to_png(label_path, png_label_path)
                # Update label_path and file_ext for further processing
                label_path = png_label_path
                file_ext = '.png'

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

            # TODO convert JPG to PNG becuase some methods don't support .jpg
            if file_ext == '.jpg':
                png_label_name = os.path.splitext(label_name)[0] + '.png'
                png_label_path = os.path.join(labelTr_path, png_label_name)
                convert_jpg_to_png(label_path, png_label_path)
                # Update label_path and file_ext for further processing
                label_path = png_label_path
                file_ext = '.png'

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
        for i in range(1, len(unique_values)):
            label_info_set['lab' + str(unique_values[i])] = i

        imageTr_list = os.listdir(imageTr_path)

        img_name = imageTr_list[0]
        img_path = os.path.join(imageTr_path, img_name)
        file_ext = os.path.splitext(img_name)[1]
        if file_ext in ['.png', '.bmp', '.tif']:
            npimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
                image_info_set[str(i)] = 'channel' + str(i)

        if file_ext == '.gz':
            file_ext = '.nii.gz'

        generate_dataset_json(os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset']), image_info_set,
                              label_info_set,
                              num_train, file_ext, dataset_name=os.environ['current_dataset'])

        dataset_id = dataset.split('_')[0].replace('Dataset', '')

        return jsonify({'status': 'Preprocessing started'})


def convert_jpg_to_png(jpg_path, png_path):
    """Converts a JPG file to PNG format."""
    with Image.open(jpg_path) as img:
        img.save(png_path, "PNG")
    os.remove(jpg_path)  # Delete the original JPG file after conversion


@app.route('/train_model', methods=['POST'])
def train_model():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})
        # fold = os.environ['current_fold']

        model_name = request.json.get('model_name', '')
        os.environ['MODEL_NAME'] = model_name
        dataset_id = request.json.get('dataset', '')
        os.environ['current_dataset'] = dataset_id

        batchSize = request.json.get('batchSize', '4')
        totalEpochs = request.json.get('totalEpochs', '100')
        learningRate = request.json.get('learningRate', '0.01')

        complete_command = f"conda activate {conda_env} && python train.py --batch_size {batchSize} --max_epochs {totalEpochs} --base_lr {learningRate}"
        batchSize = int(batchSize)
        totalEpochs = int(totalEpochs)
        learningRate = float(learningRate)

        print(complete_command)
        # Status: conda activate nnsam & & python train.py - -batch_size 4 - -max_epochs 2 - -base_lr 0.01
        # TODO convert it from command line to python code
        # train_model(batch_size=batchSize, max_epochs=totalEpochs, base_lr=learningRate)

        training_process = Process(target=run_command_async,
                                   args=("train", batchSize, totalEpochs, learningRate))
        training_process.start()
        training_process.join()
        print("Training process Ended")
        # threading.Thread(target=run_command_async, args=(complete_command,)).start()
        return jsonify({'status': complete_command})


@app.route('/run_test', methods=['POST'])
def run_test():
    global process_status
    with process_lock:
        if process_status == 'running':
            return jsonify({'error': 'A command is already running'})

    model_name = request.json.get('model_name', '')
    os.environ['MODEL_NAME'] = model_name
    dataset_id = request.json.get('dataset', '')
    os.environ['current_dataset'] = dataset_id

    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    if os.environ['MODEL_NAME'] == 'nnunet3d':
        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    val_folder = os.path.join(os.environ['nnUNet_raw'], dataset_id, f'imagesVal')
    if not os.path.exists(val_folder):
        os.makedirs(val_folder, exist_ok=True)
        src_dir_path = os.path.join(os.environ['nnUNet_raw'], dataset_id, 'imagesTr')
        src_list = os.listdir(src_dir_path)
        split_index = random_shuffle_with_spilt_index(src_list, validation_factor)
        for item in src_list[:split_index]:
            src_path = os.path.join(os.environ['nnUNet_raw'], dataset_id, 'imagesTr', item)
            dest_path = os.path.join(val_folder, item)
            shutil.copy(src_path, dest_path)

    # dataset_id = dataset_id.split('_')[0].replace('Dataset', '')
    fold = os.environ['current_fold']
    ckpt_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                             nnUNetPlans, f'fold_{fold}', 'checkpoint_final.pth')
    if os.path.exists(
            os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                         nnUNetPlans, f'fold_{fold}', 'checkpoint_best.pth')):
        shutil.copy(os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                                 nnUNetPlans, f'fold_{fold}', 'checkpoint_best.pth'), ckpt_path)

    input_folder = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
    output_folder = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                                 nnUNetPlans, 'test_pred')
    os.makedirs(output_folder, exist_ok=True)

    # if os.environ['MODEL_NAME'] == 'nnunet' or os.environ['MODEL_NAME'] == 'nnsam': complete_command = f"conda
    # activate {conda_env} && nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 2d -f {fold}"
    # elif os.environ['MODEL_NAME'] == 'nnunet3d': complete_command = f"conda activate {conda_env} &&
    # nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c 3d_fullres -f {fold}" else:

    complete_command = f"conda activate {conda_env} && python test.py"

    print(input_folder)
    print(complete_command)
    # conda activate nnsam & & python test.py
    # Namespace(max_iterations=30000, max_epochs=200, n_gpu=1, deterministic=1, base_lr=0.01, img_size=224, seed=1234,
    #           zip=False, cache_mode='part', resume=None, accumulation_steps=None, use_checkpoint=False,
    #           amp_opt_level='O1', tag=None, eval=False, throughput=False, batch_size=0)

    # TODO convert it from command line to python code
    # threading.Thread(target=run_command_async, args=(complete_command,)).start()

    test_process = Process(target=run_command_async, args=("test", 4, 2, 0.01))
    test_process.start()
    test_process.join()
    print("Test process started")

    with process_lock:
        print(input_folder)
        test_img_list_ori = os.listdir(input_folder)
        test_img_list = [test_img for test_img in test_img_list_ori]
        metric_list = []
        for test_img_name in test_img_list:
            img_path = os.path.join(input_folder, test_img_name)
            ground_truth_path = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'labelsTs',
                                             test_img_name)
            prediction_path = os.path.join(output_folder, test_img_name)

            file_ext = os.path.splitext(img_path)[1]

            data_json_file = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'dataset.json')
            with open(data_json_file, 'r') as f:
                data_json = json.load(f)
            label_num = len(data_json['labels'])

            if file_ext in ['.png', '.bmp', '.tif']:

                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
            img_with_GT = img.copy()
            for i in range(1, label_num):
                each_metric.append(calculate_metric_percase(pred == i, ground_truth == i))
                mask = np.where(pred == i, 1, 0).astype(np.uint8)
                colored_mask = np.zeros_like(img)
                colored_mask[mask == 1] = colors[i - 1]
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

                gt_mask = np.where(ground_truth == i, 1, 0).astype(np.uint8)
                colored_mask_gt = np.zeros_like(img_with_GT)
                colored_mask_gt[gt_mask == 1] = colors[i - 1]
                img_with_GT = cv2.addWeighted(img_with_GT, 1, colored_mask_gt, 0.5, 0)

                mask_result[mask == 1] = colors[i - 1]

            metric_list.append(each_metric)
            img_with_mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                                   os.environ['current_dataset'], nnUNetPlans,
                                                   'visualization_result', test_img_name)
            os.makedirs(
                os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                             nnUNetPlans, 'visualization_result'), exist_ok=True)
            cv2.imwrite(img_with_mask_save_path, img)

            img_with_GT_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                                 os.environ['current_dataset'], nnUNetPlans, 'GT_result',
                                                 test_img_name)
            os.makedirs(
                os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                             nnUNetPlans, 'GT_result'), exist_ok=True)
            cv2.imwrite(img_with_GT_save_path, img_with_GT)

            mask_save_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                          os.environ['current_dataset'], nnUNetPlans, 'mask_result', test_img_name)
            os.makedirs(
                os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                             nnUNetPlans, 'mask_result'), exist_ok=True)
            cv2.imwrite(mask_save_path, mask_result)

        metric_list = np.array(metric_list)
        dice_each_case = np.mean(metric_list[:, :, 0], axis=1)
        asd_each_case = np.mean(metric_list[:, :, 1], axis=1)

        dice_mean = np.mean(dice_each_case, axis=0)
        dice_std = np.std(dice_each_case, axis=0)

        asd_mean = np.mean(asd_each_case, axis=0)
        asd_std = np.std(asd_each_case, axis=0)

        # save into csv
        mean_csv_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'],
                                     os.environ['current_dataset'], nnUNetPlans, 'test_result_mean.csv')
        with open(mean_csv_path, 'w') as f:
            f.write('dice_mean,dice_std,asd_mean,asd_std\n')
            f.write(f'{dice_mean},{dice_std},{asd_mean},{asd_std}\n')

        csv_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                                nnUNetPlans, 'test_result.csv')
        with open(csv_path, 'w') as f:
            f.write('dice,asd\n')
            for each_metric in metric_list:
                f.write(f'{each_metric[0][0]},{each_metric[0][1]}\n')
        shutil.copy(img_with_mask_save_path, 'static/result_visiual.png')
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

    model_name = request.json.get('model_name', '')
    os.environ['MODEL_NAME'] = model_name
    dataset_id = request.json.get('dataset', '')
    os.environ['current_dataset'] = dataset_id

    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    if os.environ['MODEL_NAME'] == 'nnunet3d':
        nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    input_folder = os.path.join(os.environ['nnUNet_raw'], os.environ['current_dataset'], 'imagesTs')
    test_img_list = os.listdir(input_folder)
    method_list_path = os.path.join(os.environ['nnUNet_results'])
    method_list = os.listdir(method_list_path)

    for image_case_name in test_img_list:
        if os.environ['MODEL_NAME'] == 'nnunet3d':
            image_case_name = image_case_name + '.png'

        images = []

        for method_name in method_list:
            img_with_GT_save_path = os.path.join(os.environ['nnUNet_results'], method_name,
                                                 os.environ['current_dataset'], nnUNetPlans, 'GT_result',
                                                 image_case_name)
            print(img_with_GT_save_path)
            if os.path.exists(img_with_GT_save_path) and len(images) == 0:
                img = Image.open(img_with_GT_save_path)
                images.append(img)

            img_with_mask_save_path = os.path.join(os.environ['nnUNet_results'], method_name,
                                                   os.environ['current_dataset'], nnUNetPlans, 'visualization_result',
                                                   image_case_name)
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
            stitched_image.paste(img, (x_offset, 0))
            x_offset += img.width

        output_save_path = os.path.join(os.environ['nnUNet_results'], 'summary', os.environ['current_dataset'],
                                        'visualization_result')
        os.makedirs(output_save_path, exist_ok=True)
        stitched_image.save(os.path.join(output_save_path, image_case_name))

    # save image  into static result_visiual.png
    shutil.copy(os.path.join(output_save_path, image_case_name), 'static/result_visiual.png')

    dice_all = []
    asd_all = []
    method_exist_all = []
    for method_name in method_list:

        csv_path = os.path.join(os.environ['nnUNet_results'], method_name, os.environ['current_dataset'], nnUNetPlans,
                                'test_result_mean.csv')
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
    os.makedirs(os.path.join(os.environ['nnUNet_results'], 'summary', os.environ['current_dataset']), exist_ok=True)
    mean_csv_path = os.path.join(os.environ['nnUNet_results'], 'summary', os.environ['current_dataset'],
                                 'test_result_mean.csv')
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
    nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__2d'
    # if os.environ['MODEL_NAME'] == 'nnunet3d':
    #     nnUNetPlans = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    dir_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
                            nnUNetPlans, 'fold_' + os.environ['current_fold'])

    # if os.environ['MODEL_NAME'] == 'nnunet3d':
    #     dir_path = os.path.join(os.environ['nnUNet_results'], os.environ['MODEL_NAME'], os.environ['current_dataset'],
    #                             'nnUNetTrainer__nnUNetPlans__3d_fullres', 'fold_' + os.environ['current_fold'])

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
    global process_status
    return jsonify({'status': process_status})


@app.route('/stop_command', methods=['POST'])
def stop_command():
    global current_process
    if current_process:
        try:
            print("Terminating process...")
            # kill_process_tree(current_process.pid)
            # current_process = None
            current_process.terminate()
            current_process.join()
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

    os.makedirs(os.path.join(os.getcwd(), 'nnUNet_raw'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'nnUNet_preprocessed'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'nnUNet_results'), exist_ok=True)

    os.environ['nnUNet_raw'] = os.path.join(os.getcwd(), 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.join(os.getcwd(), 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = os.path.join(os.getcwd(), 'nnUNet_results')

    if os.environ.get('current_dataset') is None:
        os.environ['current_dataset'] = 'Dataset0'
        os.environ['MODEL_NAME'] = 'nnunet'
    if os.environ.get('current_fold') is None:
        os.environ['current_fold'] = '0'

    with open(output_file, 'w') as f:
        f.write("\n")

    webbrowser.open("http://127.0.0.1:5000/", new=2)

    app.run(debug=True, host='0.0.0.0', port=5000)

# todo list: resize, 3d, 优化去除fold选择
