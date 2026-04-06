import json
def coco_to_odvg(coco_file_path, odvg_output_path):
    """
    Converts a dataset from standard COCO JSON format to the ODVG-like 
    JSON Lines (JSONL) format.

    Args:
        coco_file_path (str): Path to the input COCO JSON file.
        odvg_output_path (str): Path to save the output ODVG JSONL file.
    """
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Create lookup dictionaries for fast access
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # A temporary dictionary to group all annotations (detections) by image ID
    image_detections = {}

    # 2. Group annotations by image_id and format the bounding boxes
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        
        # COCO bbox format: [x_top_left, y_top_left, width, height]
        x, y, w, h = ann['bbox']
        
        # ODVG bbox format: [x_min, y_min, x_max, y_max]
        bbox_odvg = [x, y, x + w, y + h]
        
        # Create the detection instance dictionary
        detection_instance = {
            "bbox": bbox_odvg,
            "label": ann['category_id'],
            "category": category_map.get(ann['category_id'], "unknown"),
            "area": ann['area']  # <--- NEW FIELD ADDED HERE
        }
        
        if img_id not in image_detections:
            image_detections[img_id] = []
        
        image_detections[img_id].append(detection_instance)
        
    # Structure the final ODVG output (list of dictionaries)
    odvg_output = []
    for img_id, detections in image_detections.items():
        # Skip images that don't exist in the image map (e.g., deleted images)
        if img_id not in image_map:
            continue
            
        img_info = image_map[img_id]
        
        # Construct the final ODVG object for the image
        odvg_entry = {
            "filename": img_info['file_name'],
            "height": img_info['height'],
            "width": img_info['width'],
            "detection": {
                "instances": detections
            }
        }
        odvg_output.append(odvg_entry)

    # 4. Save the new JSONL file
    odvg_output_path = odvg_output_path if odvg_output_path.endswith('.jsonl') else odvg_output_path + '.jsonl'
    with open(odvg_output_path, 'w') as f:
        for entry in odvg_output:
            # Write each dictionary as a single JSON object per line (JSONL format)
            f.write(json.dumps(entry) + '\n')
        
    print(f"\n Conversion complete, saved ODVG data to: {odvg_output_path}")


# Save the dummy COCO file
coco_path = 'dior_train_class_split.json'
# 2. Run the conversion
odvg_path = 'train_class_split_curated_odvg.jsonl'
coco_to_odvg(coco_path, odvg_path)
