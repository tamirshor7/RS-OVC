import json
import os
import math
from copy import deepcopy
from typing import List
from collections import defaultdict



def rescale_bbox_to_area_four(bbox_coco_format: List[int]) -> List[int]:
    """
    Rescales a COCO format bounding box (x_min, y_min, w, h) to have
    an area of 4 while maintaining the center and rounding indicators to integer.

    :param bbox_coco_format: A list [x_min, y_min, w, h] of the original box.
    :return: A list [x_min_new_int, y_min_new_int, w_new_int, h_new_int] with integer indicators.
    """
    x_min, y_min, w, h = bbox_coco_format

    # Handle zero dimensions to avoid division by zero or errors
    w = max(1, w)
    h = max(1, h)
    
    # 1. Calculate the center coordinates (cx, cy)
    center_x = x_min + w / 2
    center_y = y_min + h / 2

    # 2. Determine new float dimensions for Area = 4 (w_new * h_new = 4)
    h_new_float = 2 * math.sqrt(h / w)
    w_new_float = 2 * math.sqrt(w / h)

    # 3. Round the dimensions to the nearest integer, ensuring min size of 1
    w_new_int = max(1, round(w_new_float))
    h_new_int = max(1, round(h_new_float))

    # 4. Calculate the new x_min and y_min while preserving the center
    x_min_new_int = round(center_x - w_new_int / 2)
    y_min_new_int = round(center_y - h_new_int / 2)

    # 5. Return the new bounding box in COCO format with integer indicators
    return [x_min_new_int, y_min_new_int, w_new_int, h_new_int]


def convert_nwpu_moc_to_coco_class_split(
    annotations_root_dir: str,  
    test_classes: List[str],  # List of classes designated for the test set
    min_instance_per_class: int = 30 
) -> None:
    """
    Converts raw NWPU-MOC JSON annotation files (center points) to a COCO-like 
    format applying class-based splitting, sub-id, few-shot logic, and removing the 'others' class.
    """
    
    # Define a fixed size for the synthetic bounding box from point annotation
    BBOX_SIZE = 16  
    HALF_SIZE = BBOX_SIZE // 2

    # --- DEFINITIVE NWPU-MOC Categories ---
    MOC_CATEGORIES = [
        {"id": 0, "name": "airplane"}, {"id": 1, "name": "boat"},  
        {"id": 2, "name": "car"}, {"id": 3, "name": "container"},  
        {"id": 4, "name": "farmland"}, {"id": 5, "name": "house"},  
        {"id": 6, "name": "industrial_building"}, {"id": 7, "name": "mansion"},  
        {"id": 8, "name": "pool"}, {"id": 9, "name": "stadium"},  
        {"id": 10, "name": "tree"}, {"id": 11, "name": "truck"},  
        {"id": 12, "name": "vessel"}, {"id": 13, "name": "others"} 
    ]

    # Initialize COCO structures for final output
    base_coco_structure = {"images": [], "annotations": [], "categories": []}
    coco_train = deepcopy(base_coco_structure)
    coco_test = deepcopy(base_coco_structure)

    class_name_to_id = {}
    image_annotations_by_class = defaultdict(lambda: defaultdict(list))  
    class_split_map = {}
    
    # --- 1. Build Category Map and Split Assignment ---
    for cat in MOC_CATEGORIES:
        class_name = cat['name']
        
        # RULE: Remove "others" class entirely
        if class_name == 'others':
            continue 
            
        class_name_to_id[class_name] = cat['id']
        
        if class_name in test_classes:
            class_split_map[class_name] = 'test'
            # Add all categories to both splits' category lists
            coco_test["categories"].append(cat)
            coco_train["categories"].append(cat)
        else:
            class_split_map[class_name] = 'train'
            coco_train["categories"].append(cat)
            coco_test["categories"].append(cat)
            
    if not class_name_to_id:
        print("No valid classes found after filtering. Exiting.")
        return

    # --- 2. Load Raw Data and Consolidate by Image/Class ---
    print(f"Searching for JSON files recursively in: {annotations_root_dir}")
    
    for root, _, files in os.walk(annotations_root_dir):
        class_folder_name = os.path.basename(root)
        
        if class_folder_name in class_name_to_id:
            class_name = class_folder_name
            category_id = class_name_to_id[class_name]
            
            # The structure for NWPU-MOC is often class-folder/jsons/
            json_dir = os.path.join(root, "jsons")
            if not os.path.isdir(json_dir):
                 continue
                 
            for filename in os.listdir(json_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(json_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                            img_id = data.get("img_id")
                            #A3_2020_orth25_7_35_3.png
                            if not img_id or (img_id=="A3_2020_orth25_7_35_3.png" and class_folder_name=='truck'): #we filter out this specific image since we suspect it's truck annotation (15) is innaccurate                                
                                continue  
                            
                            # Consolidate points by (img_id, class_name)
                            for point in data.get("points", []):
                                x_center = point['x']
                                y_center = point['y']
                                
                                # Convert point to initial synthetic COCO bbox [x_min, y_min, w, h]
                                x_min = math.floor(x_center - HALF_SIZE)
                                y_min = math.floor(y_center - HALF_SIZE)
                                
                                bbox_initial = [
                                    max(0, x_min),  
                                    max(0, y_min),  
                                    BBOX_SIZE,
                                    BBOX_SIZE
                                ]
                                
                                # Store annotation temporarily with its category
                                temp_ann = {
                                    "area": BBOX_SIZE * BBOX_SIZE,
                                    "bbox": bbox_initial,
                                    "category_id": category_id,
                                    "original_img_id": img_id, 
                                    "iscrowd": 0,
                                    "segmentation": [],
                                }
                                # Store this specific instance under its image_id and class_name
                                image_annotations_by_class[img_id][class_name].append(temp_ann)
                                
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    if not image_annotations_by_class:
        print("No annotation files found or processed. Exiting.")
        return

    # --- 3. Apply Few-Shot Logic, Rescaling, and Final COCO Assembly (Class Split) ---
    
    final_train_image_id = 0
    final_test_image_id = 0
    final_train_annotation_id = 0
    final_test_annotation_id = 0
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 1024, 1024 

    # Iterate over original images
    for original_img_id, classes_data in image_annotations_by_class.items():
        
        image_sub_id = 0
        
        # Iterate over classes present in the original image
        for class_name, annotations in classes_data.items():
            
            # Skip if the class was filtered (e.g., 'others')
            if class_name not in class_split_map:
                continue

            # Filter: Only consider classes with enough instances
            if len(annotations) < min_instance_per_class:
                continue

            # Determine target split and corresponding counters/structures
            split_target = class_split_map[class_name]
            
            if split_target == 'train':
                base_image_id = final_train_image_id
                target_coco = coco_train
            else: # split_target == 'test'
                base_image_id = final_test_image_id
                target_coco = coco_test
            
            # --- Create a new image entry (one per class, tied to sub_id) ---
     
            image_dict = {
                "file_name": original_img_id, # Assuming image files are jpg
                "id": base_image_id,  
                "sub_id": image_sub_id, #category 
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "count":len(annotations)
            }
            target_coco["images"].append(deepcopy(image_dict))

            # --- Process Annotations (Few-Shot Logic) ---
            new_annotations = []
            
            for i, ann in enumerate(annotations):
                
                # Check Few-Shot Exemplar condition (i < 3)
                if i >= 3:
                    # Rescaling: Convert area to 4, keeping center/ratio
                    original_bbox = ann['bbox']
                    rescaled_bbox = rescale_bbox_to_area_four(original_bbox)
                    
                    ann['area'] = 4 
                    ann['bbox'] = rescaled_bbox
                
                # Update annotation ID and Image ID for the new COCO entry
                # Use the specific split's annotation ID counter
                if split_target == 'train':
                    current_ann_id = final_train_annotation_id
                    final_train_annotation_id += 1
                else:
                    current_ann_id = final_test_annotation_id
                    final_test_annotation_id += 1
                    
            
                ann['id'] = current_ann_id
                ann['image_id'] = base_image_id 
                ann['image_sub_id'] = image_sub_id
                
                new_annotations.append(ann)
            
            # Add the processed annotations and increment image counters
            target_coco["annotations"].extend(new_annotations)
            image_sub_id += 1 
            
            # Increment the specific split's image ID counter
            if split_target == 'train':
                final_train_image_id += 1
            else:
                final_test_image_id += 1

    # --- 4. Final Save ---

    output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, f"nwpu_train_class_split.json")
    test_file = os.path.join(output_dir, f"nwpu_test_class_split.json")

    with open(train_file, 'w') as f:
        json.dump(coco_train, f, indent=4)
        
    with open(test_file, 'w') as f:
        json.dump(coco_test, f, indent=4)
        
    print(f"\n--- Class-Split Conversion Complete ---")
    print(f"Train Images saved to: {train_file} ({len(coco_train['images'])} class-splits)")
    print(f"Test Images saved to: {test_file} ({len(coco_test['images'])} class-splits)")
    
    return

# --- EXAMPLE USAGE (Manual Setup) ---


RAW_JSON_DIRECTORY = "annotations"  

# Define the set of classes that will ONLY appear in the test set.
TEST_CLASSES = ["stadium","airplane","boat","truck","house"] 



convert_nwpu_moc_to_coco_class_split(
     annotations_root_dir=RAW_JSON_DIRECTORY,
     test_classes=TEST_CLASSES,
     min_instance_per_class=4)
