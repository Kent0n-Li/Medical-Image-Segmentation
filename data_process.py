import os
from config import *
from web import *


if __name__ == '__main__':
    # Define target paths
    target_paths = {
        'training_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTr'),
        'training_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTr'),
        'testing_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs'),
        'testing_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs'),
        'validation_image': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesVal'),
        'validation_label': os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsVal'),
    }
    try:

        #jpg to png
        for path_each in target_paths.values():
            convert_jpg_to_png_all_from_path(path_each)
            

        unique_values = find_unique_labels(target_paths['training_label'])

        convert_label_by_searchsorted(target_paths['training_label'], unique_values)
        convert_label_by_searchsorted(target_paths['validation_label'], unique_values)
        convert_label_by_searchsorted(target_paths['testing_label'], unique_values)

        # Generate dataset.json
        print_web(f"Generating dataset.json for {os.environ['current_dataset']}")
        npimg_path = os.path.join(target_paths['training_image'], os.listdir(target_paths['training_image'])[0])
        npimg = cv2.imread(npimg_path, cv2.IMREAD_UNCHANGED)

        img_channel = 3 if len(npimg.shape) == 3 else 1
        label_class_num = len(unique_values)

        dataset_id = len(os.listdir(os.environ['medseg_raw'])) + 1
        dataset_id = "{:03}".format(dataset_id)

        image_size = npimg.shape[0]

        save_dataset_json(dataset_id, os.environ['current_dataset'], image_size, img_channel, label_class_num)

    except Exception as e:
        print_web(f"Error: {e}")
        raise e