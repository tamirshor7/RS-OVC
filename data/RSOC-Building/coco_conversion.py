import json
import os
import math
import numpy as np
import scipy.io as io
import glob
import cv2
from copy import deepcopy
from typing import List, Dict
from tqdm import tqdm

# --- Replicating Necessary Functions ---

def rescale_bbox_to_area_four(bbox_coco_format: List[int]) -> List[int]:
    """
    Rescales a COCO format bounding box (x_min, y_min, w, h) to have
    an area of 4 while maintaining the center and rounding indicators to integer.
    """
    x_min, y_min, w, h = bbox_coco_format
    w = max(1, w)
    h = max(1, h)
    
    center_x = x_min + w / 2
    center_y = y_min + h / 2
    
    h_new_float = 2 * math.sqrt(h / w)
    w_new_float = 2 * math.sqrt(w / h)

    w_new_int = max(1, round(w_new_float))
    h_new_int = max(1, round(h_new_float))

    x_min_new_int = round(center_x - w_new_int / 2)
    y_min_new_int = round(center_y - h_new_int / 2)

    return [x_min_new_int, y_min_new_int, w_new_int, h_new_int]


def convert_rsoc_to_coco(
    rsoc_root_dir: str,
    output_dir: str = '.',
    fixed_bbox_size: int = 16,
    min_instance_per_class: int = 4
) -> Dict:
    """
    Converts RSOC_building point annotations from .mat files into COCO format,
    synthesizing BBoxes and applying few-shot exemplar rules.
    """
    
    # --- Setup Paths and Categories ---
    
    # Since RSOC_building is a single-class counting dataset, we only define 'building' (ID 0).
    global final_train_image_id, final_test_image_id
    global final_train_annotation_id, final_test_annotation_id
    RSOC_CATEGORY_NAME = 'building'
    RSOC_CATEGORY_ID = 0
    
    coco_categories = [{"id": RSOC_CATEGORY_ID, "name": RSOC_CATEGORY_NAME, "supercategory": "structure"}]
    
    # Define directories based on the structure you provided: "RSOC_building/building/..."
    data_subdir = 'RSOC_building/building'
    
    train_img_dir = os.path.join(rsoc_root_dir, data_subdir, 'train_data', 'images')
    test_img_dir = os.path.join(rsoc_root_dir, data_subdir, 'test_data', 'images')
    
    # Ground truth directories must match the image structure but replace 'images' with 'ground_truth'
    train_gt_dir = os.path.join(rsoc_root_dir, data_subdir, 'train_data', 'ground_truth')
    test_gt_dir = os.path.join(rsoc_root_dir, data_subdir, 'test_data', 'ground_truth')

    path_sets = [
        (train_img_dir, train_gt_dir, 'train'),
        (test_img_dir, test_gt_dir, 'test')
    ]
    
    # Initialize COCO structures
    base_structure = {"images": [], "annotations": [], "categories": coco_categories}
    coco_train = deepcopy(base_structure)
    coco_test = deepcopy(base_structure)
    coco_output_map = {'train': coco_train, 'test': coco_test}

    # Initialize counters
    final_train_image_id = 0
    final_test_image_id = 0
    final_train_annotation_id = 0
    final_test_annotation_id = 0

    HALF_SIZE = fixed_bbox_size // 2
    
    # --- Process Data Splits ---
    
    for img_dir, gt_dir, split_name in path_sets:
        
        print(f"\nProcessing {split_name} split from {img_dir}")
        
        # Determine current split's counters/pointers
        if split_name == 'train':
            final_image_id_ptr = lambda: final_train_image_id
            final_annotation_id_ptr = lambda: final_train_annotation_id
            target_coco = coco_train
        else:
            final_image_id_ptr = lambda: final_test_image_id
            final_annotation_id_ptr = lambda: final_test_annotation_id
            target_coco = coco_test

        img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        
        for img_path in tqdm(img_paths, desc=f"Converting {split_name}"):
            
            # Determine GT path and load coordinates
            img_filename = os.path.basename(img_path)
            img_base = img_filename.replace('.jpg', '')
            # The GT file name convention from the example script:
            # IMG_1.jpg -> GT_IMG_1.mat (replacing IMG_ with GT_IMG_ and .jpg with .mat)
            gt_filename = img_filename.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found for {img_filename} at {gt_path}. Skipping.")
                continue

            # Load Image Dims
            img = cv2.imread(img_path)
            if img is None: continue
            height, width = img.shape[:2]

            # Load Annotations 
            try:
                mat = io.loadmat(gt_path)
                # The data is extracted from mat['center'][0,0] as per the source script.
                gt_points = mat['center'][0, 0] 
                
                # Check if points exist; mat['center'][0,0] might be an empty array
                if gt_points.size == 0 or gt_points.shape[0] == 0:
                    continue # No objects to count

                # Points must be sorted to apply few-shot indexing consistently
                # We sort by Y coordinate (row) then X (col)
                gt_points = gt_points[gt_points[:, 1].argsort()]

            except Exception as e:
                print(f"Error reading {gt_path}: {e}. Skipping.")
                continue

            # Apply Few-Shot Logic (Synthetic BBoxes)
            
            raw_annotations = []
            
            for i, pt in enumerate(gt_points):
                # Coordinates are (X, Y) where X=gt[i][0] and Y=gt[i][1]
                x_center = int(pt[0])
                y_center = int(pt[1])
                
                # Synthesize 16x16 COCO BBox format: [x_min, y_min, w, h]
                x_min = max(0, x_center - HALF_SIZE)
                y_min = max(0, y_center - HALF_SIZE)
                
                bbox_initial = [x_min, y_min, fixed_bbox_size, fixed_bbox_size]
                area = fixed_bbox_size * fixed_bbox_size
                
                # Few-Shot Rescaling (only for instances i >= 3)
                if i >= 3:
                    rescaled_bbox = rescale_bbox_to_area_four(bbox_initial)
                    area = 4 # Fixed area of 4
                    bbox_initial = rescaled_bbox
                
                # Assign ID counters and required fields
                # Use the current split's annotation ID counter
                current_ann_id = final_annotation_id_ptr()
                
                ann = {
                    "id": current_ann_id,
                    "image_id": final_image_id_ptr(), # This will be the base_image_id assigned later
                    "category_id": RSOC_CATEGORY_ID,
                    "bbox": bbox_initial,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                    # Since this is a single-class dataset, sub_id will always be 0
                    "image_sub_id": 0 
                }
                raw_annotations.append(ann)
                
                # Increment the split's annotation counter
                if split_name == 'train':
                
                    final_train_annotation_id += 1
                else:
    
                    final_test_annotation_id += 1
            
            # --- Final COCO Entry Assembly ---
            
            # Check if we have enough instances after potential point filtering (if implemented)
            if len(raw_annotations) < min_instance_per_class:
                continue

            # Assign Base Image ID (Current split's image counter)
            base_image_id = final_image_id_ptr()

            # Create new COCO Image Entry
            image_dict = {
                "file_name": img_filename, # e.g., IMG_1.jpg
                "id": base_image_id, 
                "sub_id": 0, # Always 0 for single-class RSOC_building
                "width": width,
                "height": height,
                "count":len(raw_annotations)
            }
            target_coco["images"].append(deepcopy(image_dict))

            # Update annotation image_id fields and add to final list
            for ann in raw_annotations:
                ann['image_id'] = base_image_id
                target_coco["annotations"].append(ann)

            # Increment the split's image ID counter
            if split_name == 'train':
          
                final_train_image_id += 1
            else:
                
                final_test_image_id += 1


    # --- Final Save ---
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "rsoc_train.json")
    test_file = os.path.join(output_dir, "rsoc_test.json")

    with open(train_file, 'w') as f:
        json.dump(coco_train, f, indent=4)
    with open(test_file, 'w') as f:
        json.dump(coco_test, f, indent=4)
        
    print(f"\n--- RSOC Conversion Complete ---")
    print(f"Train split saved to: {train_file} ({len(coco_train['images'])} images, {len(coco_train['annotations'])} buildings)")
    print(f"Test split saved to: {test_file} ({len(coco_test['images'])} images, {len(coco_test['annotations'])} buildings)")
    
    return {'train': coco_train, 'test': coco_test}

# --- EXAMPLE USAGE (Manual Setup) ---

# The data root should contain the RSOC building data
RSOC_DATA_ROOT = '.' 

convert_rsoc_to_coco(
    rsoc_root_dir=RSOC_DATA_ROOT,
    output_dir='.',
    fixed_bbox_size=16,
    min_instance_per_class=4
)