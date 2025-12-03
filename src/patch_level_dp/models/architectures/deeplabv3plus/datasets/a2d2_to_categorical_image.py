import glob
import json
import os
from pathlib import Path
import functools

import multiprocessing as mp
import cv2
import numpy as np
from tqdm import tqdm
from itertools import repeat

from test_dp.DeepLabV3Plus.datasets.a2d2 import TRAIN_SESSIONS, VAL_SESSIONS, TEST_SESSIONS


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # todo: grey/colour switch
    return image


def write_image(img_name, image):
    os.makedirs(str(Path(img_name).parent), exist_ok=True)
    cv2.imwrite(img_name, image)


def process_single_label(label_path, reduced_mapping):
    """
    Loads an RGB label image, converts it to a categorical format using the provided
    mapping, and saves the result. This is a top-level function to allow pickling.
    """
    save_path = label_path.replace("/label/", "/categorical_label/")
    if os.path.exists(save_path):
        return

    img = load_image(label_path)
    
    hexa = np.vectorize("{:02x}".format)
    image_1dhexa = np.sum(hexa(img).astype(object), axis=-1)
    
    categorical = np.vectorize(lambda x: reduced_mapping.get(x, 255))(image_1dhexa).astype(np.uint8)
    
    write_image(save_path, categorical)
    print(f"Processed: {label_path}")


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)

    data_path="/nfs/students/duk/camera_lidar_semantic"
    
    class_list_file = os.path.join(data_path, "class_list.json")
    with open(class_list_file, "r") as f:
        class_colour_mapping = json.load(f)

    class_group_file = os.path.join(data_path, "class_group_mapping.json")
    with open(class_group_file, "r") as f:
        groups_class_mapping = json.load(f)

    group_class_name_to_id = {k: i for i, k in enumerate(groups_class_mapping.keys())}
    class_groups_mapping = {
        category_id: group_class_name_to_id[class_group]
        for class_group, classes in groups_class_mapping.items()
        for category_id in classes
    }
    colour_to_class_group_id = {
        k: class_groups_mapping[v] for k, v in class_colour_mapping.items()
    }
    reduced_mapping = {k.replace("#", ""): v for k, v in colour_to_class_group_id.items()}
    
    all_sessions = TRAIN_SESSIONS + VAL_SESSIONS + TEST_SESSIONS
    
    print("Searching for files in specified sessions...")
    file_list = []
    for session in all_sessions:
        session_path = os.path.join(data_path, session)
        glob_path = f"{session_path}/camera/cam_*/*.png"
        image_files = glob.glob(glob_path)
        
        label_paths = [p.replace('/camera/', '/label/').replace('_camera_', '_label_') for p in image_files]
        file_list.extend(label_paths)
        
    print(f"Found {len(file_list)} files to process.")
    
    if file_list:
        print("Starting processing pool...")
    with mp.Pool(20) as p:
            args = zip(file_list, repeat(reduced_mapping))
            p.starmap(process_single_label, args, chunksize=100)
        print("Processing finished.")
    else:
        print("Warning: No files found to process. Please check the data_path.")