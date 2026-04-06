import os
import math 
from copy import deepcopy
import cv2
from tqdm import tqdm
import json
import xml.dom.minidom
import collections
from typing import List

DATA_ROOT = '.'
ANN_SUBDIR = 'Annotations/Horizontal Bounding Boxes/'
IMG_SUBDIR_TRAINVAL = 'JPEGImages-trainval/'
IMG_SUBDIR_TEST = 'JPEGImages-test/'


def group_and_filter_dicts(data_list, min_count=4):
    """
    Groups a list of dictionaries by their 'category_id' and returns only the 
    groups that contain 'min_count' or more elements.
    """
    grouped_data = collections.defaultdict(list)
    for item in data_list:
        category = item.get('category_id')
        if category is not None:
            grouped_data[category].append(item)

    filtered_groups = []
    for category_items in grouped_data.values():
        if len(category_items) >= min_count:
            filtered_groups.append(category_items)
            
    return filtered_groups


def rescale_bbox_to_area_four(bbox_coco_format):
    """
    Rescales a COCO format bounding box (x_min, y_min, w, h) to have
    an area of 4 (or closest integer area) while maintaining the same center 
    and aspect ratio, then rounds all indicators to the nearest integer.
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


class DiorCocoClassSplitConverter:
    def __init__(self, raw_data_path: str, output_json_path: str, test_classes: List[str], remove_class: str = 'vehicle'):
        self.raw_data_path = raw_data_path
        self.output_json_path = output_json_path
        self.test_classes = set(test_classes)
        self.remove_class = remove_class

        # Original DIOR categories and IDs
        self.dior_labels = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                     'chimney', 'expressway-service-area', 'expressway-toll-station',
                     'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                     'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                     'windmill')
        
        # Build the final consistent category list and split map
        self.categories_map = {}
        self.categories_list = []
        self.class_split_map = {}
        
        for idx, name in enumerate(self.dior_labels):
            if name == self.remove_class:
                continue
                
            self.categories_map[name] = idx
            self.categories_list.append({"id": idx, "name": name, "supercategory": "object"})

            
            if name == 'harbor': #we use DIOR's harbor for architecture validation
                self.class_split_map[name] = 'val'
            elif name in self.test_classes:
                self.class_split_map[name] = 'test'
            else:
                self.class_split_map[name] = 'train'


    def convert(self, min_instance_per_class: int = 4):
        
        # --- Identify all unique image IDs across trainval and test folders ---
        
        # DIOR is split into two image folders based on the original test split
        
        image_data_sources = []
        
        # Collect IDs from the trainval set
        trainval_img_path = os.path.join(self.raw_data_path, IMG_SUBDIR_TRAINVAL)
        trainval_id_file = os.path.join(self.raw_data_path, 'ImageSets/Main/trainval.txt')
        
        if os.path.exists(trainval_id_file):
            with open(trainval_id_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                image_data_sources.extend([ (img_id, trainval_img_path) for img_id in ids ])
        
        # Collect IDs from the test set
        test_img_path = os.path.join(self.raw_data_path, IMG_SUBDIR_TEST)
        test_id_file = os.path.join(self.raw_data_path, 'ImageSets/Main/test.txt')
        
        if os.path.exists(test_id_file):
            with open(test_id_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                image_data_sources.extend([ (img_id, test_img_path) for img_id in ids ])

        if not image_data_sources:
             print("Error: Could not find image ID lists (trainval.txt or test.txt). Check DATA_ROOT.")
             return
        
        # --- Initialize COCO structures for final output ---
        
        base_structure = {"images": [], "annotations": [], "categories": self.categories_list}
        coco_train = deepcopy(base_structure)
        coco_test = deepcopy(base_structure)
   
        coco_val = deepcopy(base_structure)
        
        final_train_image_id = 0
        final_test_image_id = 0
        
        final_val_image_id = 0
        
        final_train_annotation_id = 0
        final_test_annotation_id = 0
        final_val_annotation_id = 0
        
        annotations_dir = os.path.join(self.raw_data_path, ANN_SUBDIR)
        
        # --- Process Images and Apply Class Split/Few-Shot Logic ---
        
        for name, image_source_path in tqdm(image_data_sources, desc="Processing DIOR Images"):
            
            # 3.1 Get image dimensions
            image_file_path = os.path.join(image_source_path, name + ".jpg")
            img = cv2.imread(image_file_path)
            if img is None:
                # Handle cases where image files are missing
                continue 
                
            height, width = img.shape[:2]
            
            # Parse XML for annotations
            ann_list_by_class = collections.defaultdict(list)
            
            xml_path = os.path.join(annotations_dir, name + ".xml")
            if not os.path.exists(xml_path):
                continue
                
            DOMTree = xml.dom.minidom.parse(xml_path)
            objects = DOMTree.documentElement.getElementsByTagName("object")
            
            for object_ in objects:
                obj_name = object_.getElementsByTagName('name')[0].childNodes[0].data.lower()
               
                # Filter out the removed class
                if obj_name == self.remove_class or obj_name not in self.categories_map:
                    continue

                bndbox = object_.getElementsByTagName('bndbox')[0]
                xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)

                w = xmax - xmin
                h = ymax - ymin
                category_id = self.categories_map[obj_name]

                # Store the raw annotation data
                ann_list_by_class[obj_name].append({
                    "area": w * h,
                    "bbox": [xmin, ymin, w, h],
                    "category_id": category_id,
                    "original_img_name": name # Keep original name for debugging
                })
            
            # Apply Class Split and Few-Shot Logic
            
            image_sub_id = 0
            
            for class_name, raw_annotations in ann_list_by_class.items():
                
                # filter out classes with insufficient instances
                if len(raw_annotations) < min_instance_per_class:
                    continue
                
                # Determine the target split and counters based on class name
                split_target = self.class_split_map[class_name]
                
                if split_target == 'train':
                    base_image_id = final_train_image_id
                    target_coco = coco_train
                elif split_target == 'val': 
                    base_image_id = final_val_image_id
                    target_coco = coco_val
                else:
                    base_image_id = final_test_image_id
                    target_coco = coco_test

                # --- Create new COCO Image Entry ---
                image_dict = {
                    "file_name": name + ".jpg",
                    "id": base_image_id, 
                    "sub_id": image_sub_id,
                    "width": width,
                    "height": height,
                    "count":len(raw_annotations)
                }
                target_coco["images"].append(deepcopy(image_dict))

                # --- Process Annotations (Few-Shot Rescaling) ---
                new_annotations = []
                
                for i, ann in enumerate(raw_annotations):
                    
                    # Apply few-shot rescaling for i >= 3
                    if i >= 3:
                        ann['area'] = 4
                        ann['bbox'] = rescale_bbox_to_area_four(ann['bbox'])

                    # Assign final unique IDs for the specific split
                    if split_target == 'train':
                        current_ann_id = final_train_annotation_id
                        final_train_annotation_id += 1
                    elif split_target == 'val': 
                        current_ann_id = final_val_annotation_id
                        final_val_annotation_id += 1
                    else:
                        current_ann_id = final_test_annotation_id
                        final_test_annotation_id += 1
                        
                    ann['id'] = current_ann_id
                    ann['image_id'] = base_image_id
                    ann['image_sub_id'] = image_sub_id
                    ann['iscrowd'] = 0
                    ann['segmentation']: []
                    
                    new_annotations.append(ann)

                target_coco["annotations"].extend(new_annotations)
                image_sub_id += 1
                
                # Increment the split-specific image ID counter
                if split_target == 'train':
                    final_train_image_id += 1
                elif split_target == 'val': 
                    final_val_image_id += 1
                else:
                    final_test_image_id += 1

        
        os.makedirs(self.output_json_path, exist_ok=True)
        
        train_file = os.path.join(self.output_json_path, "dior_train_class_split.json")
        val_file = os.path.join(self.output_json_path, "dior_val_class_split.json") 
        test_file = os.path.join(self.output_json_path, "dior_test_class_split.json")

        with open(train_file, 'w') as f:
            json.dump(coco_train, f, indent=4)
       
        with open(val_file, 'w') as f:
            json.dump(coco_val, f, indent=4)
        with open(test_file, 'w') as f:
            json.dump(coco_test, f, indent=4)
            
        print(f"\n--- DIOR Class-Split Conversion Complete ---")
        print(f"Train Classes: {set(self.dior_labels) - self.test_classes - {self.remove_class} - {'harbor'}}")
        print(f"Val Classes: {{'harbor'}}")
        print(f"Test Classes: {self.test_classes}")
        print(f"Train split saved to: {train_file} ({len(coco_train['images'])} class-splits, {len(coco_train['annotations'])} annotations)")
        print(f"Val split saved to: {val_file} ({len(coco_val['images'])} class-splits, {len(coco_val['annotations'])} annotations)")
        print(f"Test split saved to: {test_file} ({len(coco_test['images'])} class-splits, {len(coco_test['annotations'])} annotations)")


# --- EXAMPLE USAGE ---

DIOR_TEST_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "bridge",
    "chimney",
    "dam",
    "stadium",
    "storagetank",
    "windmill"
    ]

                   

converter = DiorCocoClassSplitConverter(
    raw_data_path=DATA_ROOT,
    output_json_path=".",
    test_classes=DIOR_TEST_CLASSES,
    remove_class=None
)

# Run the conversion
converter.convert(min_instance_per_class=4)