import json
import os
import math
import cv2
import glob
from copy import deepcopy
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

# --- Global DOTA Categories (Used for consistent ID mapping) ---
DOTA_LABELS = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank', 
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    'container', 'airport', 'helipad', 'chimney', 'expressway-service-area', 
    'expressway-toll-station', 'dam', 'golffield', 'overpass', 'train-station', 
    'wind-mill', 'container-crane'
)
DOTA_CLS2LBL = {k: v for v, k in enumerate(DOTA_LABELS)}


# --- Counters (Initialized here for module-level scope) ---
final_train_image_id = 0
final_test_image_id = 0
final_val_image_id = 0  # Added val counter

final_train_annotation_id = 0
final_test_annotation_id = 0
final_val_annotation_id = 0 # Added val counter


# --- Helper Functions (Rescaling and OBB-to-AABB) ---

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


def obb_to_aabb(obb_coords: List[float]) -> List[int]:
    """
    Converts 8 OBB coordinates into COCO-style Axis-Aligned Bounding Box (AABB) 
    [x_min, y_min, w, h].
    """
    xs = [obb_coords[i] for i in range(0, 8, 2)]
    ys = [obb_coords[i] for i in range(1, 8, 2)]
    
    x_min = math.floor(min(xs))
    y_min = math.floor(min(ys))
    x_max = math.ceil(max(xs))
    y_max = math.ceil(max(ys))
    
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)

    return [x_min, y_min, w, h]


def parse_dota_annotation(annotation_path: str) -> List[Dict]:
    """
    Parses a single DOTA text annotation file and returns a list of raw object dicts.
    """
    raw_annotations = []
    
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    # skip header lines (imagesource, gsd)
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('imagesource', 'gsd'))]

    for line in data_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        
        obb_coords = [float(p) for p in parts[:8]]
        class_name = parts[8].lower()
        
        if class_name not in DOTA_CLS2LBL:
            continue
            
        category_id = DOTA_CLS2LBL[class_name]
        
        bbox_coco = obb_to_aabb(obb_coords)
        
        raw_annotations.append({
            "bbox": bbox_coco,
            "category_id": category_id,
            "area": bbox_coco[2] * bbox_coco[3],
            "iscrowd": 0,
        })
        
    return raw_annotations


def convert_dota_to_coco_class_split(
    dota_root_dir: str,
    test_classes: List[str],  # List of classes designated for the test set
    output_dir: str = '.',
    min_instance_per_class: int = 4
) -> Dict:
    """
    Converts DOTA annotations to COCO format, applying a class-based split 
    and few-shot exemplar rescaling logic.
    """
    
    # --- Setup COCO Structures and Split Mapping ---
    
    # Declare counters as global for modification
    global final_train_image_id, final_test_image_id, final_val_image_id
    global final_train_annotation_id, final_test_annotation_id, final_val_annotation_id
    
    # Create final COCO categories list
    coco_categories = [
        {"id": DOTA_CLS2LBL[name], "name": name, "supercategory": "aerial"}
        for name in DOTA_LABELS
    ]
    
    base_structure = {"images": [], "annotations": [], "categories": coco_categories}
    coco_train = deepcopy(base_structure)
    coco_test = deepcopy(base_structure)
    coco_val = deepcopy(base_structure) # Init val structure
    
    # --- Updated Class Split Logic ---
    class_split_map = {}
    validation_classes = ['harbor', 'helicopter'] # Hardcoded per paper specs
    
    for name in DOTA_LABELS:
        if name in validation_classes:
             class_split_map[name] = 'val'
        elif name in test_classes:
             class_split_map[name] = 'test'
        else:
             class_split_map[name] = 'train'

    # Define directories to search (combine train and val for class-wise splitting)
    split_info = {
        'train': {
            'ann_dir': os.path.join(dota_root_dir, 'train', 'labelTxt-v1.0'),
            'img_dir': os.path.join(dota_root_dir, 'train', 'images', 'images'),
        },
        'val': {
            'ann_dir': os.path.join(dota_root_dir, 'val', 'labelTxt-v1.0'),
            'img_dir': os.path.join(dota_root_dir, 'val', 'images', 'images'),
        }
    }
    
    # Process All Annotation Files ---
    all_ann_paths = []
    for paths in split_info.values():
        all_ann_paths.extend(glob.glob(os.path.join(paths['ann_dir'], '*.txt')))
    

    for ann_path in tqdm(all_ann_paths, desc="Processing All DOTA Annotations"):
        
        # Determine the source split based on the annotation path
        if 'train' in ann_path:
            source_split = 'train'
        elif 'val' in ann_path:
            source_split = 'val'
        else:
            continue
            
        paths = split_info[source_split]
        
        # Get file names and dimensions
        img_base_name = os.path.basename(ann_path).replace('.txt', '')
        img_file_name = img_base_name + '.png'
        img_path = os.path.join(paths['img_dir'], img_file_name)

        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None: continue # Safety check
        height, width = img.shape[:2]

        # Parse and group annotations
        raw_annotations = parse_dota_annotation(ann_path)
        
        annotations_by_class = defaultdict(list)
        for ann in raw_annotations:
            class_name = DOTA_LABELS[ann['category_id']]
            annotations_by_class[class_name].append(ann)

        image_sub_id = 0
        
        # --- Apply Few-Shot Rules and Create COCO Entries ---
        
        for class_name, annotations in annotations_by_class.items():
            
            # Determine final split target based on class name
            split_target = class_split_map.get(class_name, 'train') 
            
            # Filter: Only consider classes with enough instances
            if len(annotations) < min_instance_per_class:
                continue

            # Determine split-specific resources
            if split_target == 'train':
                base_image_id = final_train_image_id
                target_coco = coco_train
                final_img_id_ptr = 'final_train_image_id'
                apply_fs = True
            elif split_target == 'val': # Val selection
                base_image_id = final_val_image_id
                target_coco = coco_val
                final_img_id_ptr = 'final_val_image_id'
                apply_fs = True
            else: # 'test'
                base_image_id = final_test_image_id
                target_coco = coco_test
                final_img_id_ptr = 'final_test_image_id'
                apply_fs = True
            
            # --- Create COCO Image Entry (Sub-ID Split) ---
            image_dict = {
                "file_name": img_file_name,
                "id": base_image_id, 
                "sub_id": image_sub_id, 
                "width": width,
                "height": height,
                "count":len(annotations)
            }
            target_coco["images"].append(deepcopy(image_dict))

            # Process and assign annotation IDs
            for i, ann in enumerate(annotations):
                
                # Few-Shot Rescaling (i >= 3)
                if apply_fs and i >= 3:
                    ann['bbox'] = rescale_bbox_to_area_four(ann['bbox'])
                    ann['area'] = 4
                
                # Assign and increment annotation ID
                if split_target == 'train':
                    ann['id'] = final_train_annotation_id
                    globals()['final_train_annotation_id'] += 1
                elif split_target == 'val': # Val increment
                    ann['id'] = final_val_annotation_id
                    globals()['final_val_annotation_id'] += 1
                else: # test
                    ann['id'] = final_test_annotation_id
                    globals()['final_test_annotation_id'] += 1
                    
                ann['image_id'] = base_image_id
                ann['image_sub_id'] = image_sub_id
                
                target_coco["annotations"].append(ann)

            image_sub_id += 1 
            
            # Increment global image counter
            globals()[final_img_id_ptr] += 1

    # --- Final Save ---
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "dota_train_class_split.json")
    val_file = os.path.join(output_dir, "dota_val_class_split.json") # Val file path
    test_file = os.path.join(output_dir, "dota_test_class_split.json")

    with open(train_file, 'w') as f:
        json.dump(coco_train, f, indent=4)
    with open(val_file, 'w') as f: # Save val
        json.dump(coco_val, f, indent=4)
    with open(test_file, 'w') as f:
        json.dump(coco_test, f, indent=4)
        
    print(f"\n--- DOTA Class-Split Conversion Complete ---")
    print(f"Train split saved to: {train_file} ({len(coco_train['images'])} images)")
    print(f"Val split saved to: {val_file} ({len(coco_val['images'])} images)")
    print(f"Test split saved to: {test_file} ({len(coco_test['images'])} images)")
    
    return {'train': coco_train, 'val': coco_val, 'test': coco_test}



DOTA_ROOT = "." 
TEST_CLASSES = ['plane', 'baseball-diamond', 'bridge', 'storage-tank', 
    'soccer-ball-field', 'roundabout',
    'container', 'airport', 'chimney', 
    'dam',
    'wind-mill', 'container-crane'] 

convert_dota_to_coco_class_split(
    dota_root_dir=DOTA_ROOT,
    test_classes=TEST_CLASSES,
    output_dir='.',
    min_instance_per_class=4
)
