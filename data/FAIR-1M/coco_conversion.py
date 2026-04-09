import json
import os
import math
from copy import deepcopy
from typing import List, Dict
import pandas as pd
import argparse


# The rescale_bbox_to_area_four function remains exactly the same.
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


def get_consolidated_name(original_name: str) -> str:
    """
    Consolidates fine-grained FAIR1M classes into coarser categories.
    """
    CONSOLIDATED_MAP = {
        # Ships
        'Warship': 'ship', 'Dry Cargo Ship': 'ship', 'Liquid Cargo Ship': 'ship', 
        'other-ship': 'ship', 'Engineering Ship': 'ship', 'Passenger Ship': 'ship',
        
        # Boats
        'Tugboat': 'boat', 'Motorboat': 'boat', 'Fishing Boat': 'boat', 
        
        # Trucks
        'Cargo Truck': 'truck', 'Dump Truck': 'truck', 'Truck Tractor': 'tractor',
        
        # Cars/Vans/Buses/Other Vehicles
        'Small Car': 'car', 'Van': 'van', 'Bus': 'vehicle', 'Excavator': 'excavator', 
        'other-vehicle': 'vehicle', 'Trailer': 'van', 'Tractor': 'tractor',
        
        # Airplanes
        'A220': 'airplane', 'other-airplane': 'airplane', 'Boeing787': 'airplane', 
        'A321': 'airplane', 'Boeing737': 'airplane', 'A330': 'airplane', 
        'Boeing777': 'airplane', 'Boeing747': 'airplane', 'A350': 'airplane', 
        'ARJ21': 'airplane', 'C919': 'airplane', 
        
        # Fields/Courts/Others 
        'Baseball Field': 'baseball field', 'Tennis Court': 'tennis court', 
        'Basketball Court': 'basketball court', 'Football Field': 'football field',
        
        # Roads/Structures
        'Intersection': 'intersection', 'Bridge': 'bridge', 'Roundabout': 'roundabout',
    }
    
    # If the class name isn't found, keep it as is (though this shouldn't happen
    # if the input is a known FAIR1M class).
    return CONSOLIDATED_MAP.get(original_name, original_name)


def convert_fair1m_to_coco_class_split(
    labels_parquet_path: str,
    test_classes: List[str], # List of consolidated class names for testing
    output_dir: str = '.',
    min_instance_per_class: int = 4
) -> Dict:
    """
    Converts FAIR1M dataset to COCO format, consolidates fine-grained classes,
    and performs a class-based split (Train/Test).
    """
    
    # --- 1. Load Data and Prepare Consolidated Category Mapping ---
    try:
        df = pd.read_parquet(labels_parquet_path)
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return {}

    # Apply consolidation to create a new column for grouping/mapping
    df['Consolidated_Category'] = df['Category'].apply(get_consolidated_name)
    
    unique_categories = df['Consolidated_Category'].unique()
    
    # Map class name to ID (0-based indexing)
    category_mapping = {name: i for i, name in enumerate(unique_categories)} 
    
    coco_categories = [
        {"id": cat_id, "name": cat_name, "supercategory": "object"}
        for cat_name, cat_id in category_mapping.items()
    ]

    # Initialize COCO structures for final output
    base_structure = {"images": [], "annotations": [], "categories": coco_categories}
    coco_train = deepcopy(base_structure)
    coco_test = deepcopy(base_structure)
    
    # Determine which class goes to which split
    class_split_map = {name: ('test' if name in test_classes else 'train') for name in unique_categories}

    # --- 2. Process Labels and Apply Few-Shot Logic (Class Split) ---

    final_train_image_id = 0
    final_test_image_id = 0
    final_train_annotation_id = 0
    final_test_annotation_id = 0
    
    # Group ALL data by file path, ignoring the original 'Split' column
    # Grouping by 'FilePath' is correct for ensuring all objects in one image are processed together
    for filepath, img_data in df.groupby('FilePath'):
        
        # Get image dimensions from the first row of the image's data
        img_row = img_data.iloc[0]
        width = int(img_row['ImageWidth'])
        height = int(img_row['ImageHeight'])
        
        # Extract file name without the path
        # Assuming the final part of FilePath is the unique identifier (e.g., '12345.jpg')
        file_name = filepath.split("/")[-1]
        
        image_sub_id = 0
        
        # Iterate over classes present in this original image (using the CONSOLIDATED name)
        for class_name, annotations_df in img_data.groupby('Consolidated_Category'):
            
            annotations = annotations_df.to_dict('records')

            # Determine split and filter out unknown/unwanted classes
            if class_name not in class_split_map:
                continue
                
            split_target = class_split_map[class_name]
            
            # FILTER IMAGES WITH LESS THAN MIN_INSTANCE_PER_CLASS
            if len(annotations) < min_instance_per_class:
                continue

            # Determine target split and corresponding counters/structures
            if split_target == 'train':
                base_image_id = final_train_image_id
                target_coco = coco_train
            else: # split_target == 'test'
                base_image_id = final_test_image_id
                target_coco = coco_test

            # --- Create a new image entry (one per class, tied to sub_id) ---
            image_dict = {
                "file_name": file_name, 
                "id": base_image_id, 
                "sub_id": image_sub_id, 
                "width": width,
                "height": height,
                "count":len(annotations)
            }
            target_coco["images"].append(deepcopy(image_dict))

            # --- Process Annotations ---
            new_annotations = []
            category_id = category_mapping[class_name]
            
            # Apply few-shot logic only to the Train split (common practice)
            apply_few_shot_logic = True#(split_target == 'train')
            
            for i, ann_row in enumerate(annotations):
                
                # Convert AABB to COCO format (x_min, y_min, w, h)
                x_min = int(ann_row['x_min'])
                y_min = int(ann_row['y_min'])
                w = max(1, int(ann_row['x_max'] - x_min))
                h = max(1, int(ann_row['y_max'] - y_min))
                
                bbox_initial = [x_min, y_min, w, h]
                area = w * h
                
                # Apply few-shot rescaling for the training set's non-exemplar instances
                if apply_few_shot_logic and i >= 3:
                    rescaled_bbox = rescale_bbox_to_area_four(bbox_initial)
                    area = 4 # Fixed area of 4
                    bbox_initial = rescaled_bbox
                
                # Assign final unique IDs for the specific split
                if split_target == 'train':
                    current_ann_id = final_train_annotation_id
                    final_train_annotation_id += 1
                else:
                    current_ann_id = final_test_annotation_id
                    final_test_annotation_id += 1
                
                # Create the final annotation entry
                ann = {
                    "id": current_ann_id,
                    "image_id": base_image_id, 
                    "category_id": category_id,
                    "bbox": bbox_initial,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                    "image_sub_id": image_sub_id 
                }
                
                new_annotations.append(ann)
            
            # Add annotations and increment counters
            target_coco["annotations"].extend(new_annotations)
            image_sub_id += 1 
            
            # Increment the specific split's image ID counter
            if split_target == 'train':
                final_train_image_id += 1
            else:
                final_test_image_id += 1

    # --- 3. Final Save ---
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, f"fair1m_train_class_split.json")
    test_file = os.path.join(output_dir, f"fair1m_test_class_split.json")

    with open(train_file, 'w') as f:
        json.dump(coco_train, f, indent=4)
    # The original 'Val' split logic is removed; only train and test are outputted.
    with open(test_file, 'w') as f:
        json.dump(coco_test, f, indent=4)
        
    print(f"\n--- FAIR1M Class-Split Conversion Complete ---")
    print(f"Total Consolidated Categories: {len(category_mapping)}")
    print(f"Train split saved to: {train_file} ({len(coco_train['images'])} images, {len(coco_train['annotations'])} annotations)")
    print(f"Test split saved to: {test_file} ({len(coco_test['images'])} images, {len(coco_test['annotations'])} annotations)")
    
    return {'train': coco_train, 'test': coco_test}

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fair1m_root",
        type=str,
        required=True,
        help="Path to FAIR-1M dataset root containing labels.parquet"
    )

    parser.add_argument(
        "--min_instances",
        type=int,
        default=4,
        help="Minimum instances per class per image"
    )

    parser.add_argument(
        "--test_classes",
        nargs="+",
        default=["truck", "boat", "airplane", "baseball field", "tractor"],
        help="Consolidated classes assigned to the test split"
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_file = os.path.join(args.fair1m_root, "labels.parquet")

    convert_fair1m_to_coco_class_split(
        labels_parquet_path=labels_file,
        test_classes=args.test_classes,
        output_dir=script_dir,
        min_instance_per_class=args.min_instances
    )