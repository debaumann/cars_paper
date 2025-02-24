#!/usr/bin/env python3
import csv
import json
import os
import ast

def parse_annotations(annotations_path, mapping_dict, subject_key, output_path, accepted_labels):
    """
    Reads the annotations CSV and writes a new CSV with:
      - address: <subject_key>/<video_folder>/0/
      - start_frame: computed from the first temporal coordinate * 30
      - end_frame: computed from the second temporal coordinate * 30 (or same as start_frame if absent)
      - numeric_label: looked up from mapping_dict using the metadata label
      
    Only annotations with a label that is in accepted_labels are processed.
    The CSV is expected to have columns:
      metadata_id, file_list, flags, temporal_coordinates, spatial_coordinates, metadata
    and any lines starting with '#' are skipped.
    """
    output_rows = []
    
    # Read file lines, ignoring comment lines starting with "#"
    with open(annotations_path, 'r', newline='') as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    
    reader = csv.reader(lines)
    for row in reader:
        if len(row) < 6:
            continue  # Skip invalid entries

        # Unpack fields (order per sample)
        _, file_list_str, _, temporal_str, _, metadata_str = row

        # Parse file_list (a JSON list of filenames)
        try:
            file_list = json.loads(file_list_str)
        except json.JSONDecodeError:
            file_list = ast.literal_eval(file_list_str)
        if not file_list:
            continue
        video_filename = file_list[0]
        # Remove the .mp4 extension to get the folder name
        folder_name, _ = os.path.splitext(video_filename)
        # Build the address as: <subject_key>/<folder_name>/0/
        address = f"{subject_key}/{folder_name}/0/"

        # Parse temporal_coordinates (expects a JSON list, e.g. "[1.7105,2.48134]")
        try:
            temporal_coords = json.loads(temporal_str)
        except json.JSONDecodeError:
            temporal_coords = ast.literal_eval(temporal_str)
        if not temporal_coords:
            continue
        # Compute the start frame (assuming 30 fps)
        start_time = float(temporal_coords[0])
        start_frame = int(start_time * 30)
        # If a second coordinate exists, compute the end frame; otherwise, use start_frame.
        if len(temporal_coords) > 1:
            end_time = float(temporal_coords[1])
            end_frame = int(end_time * 30)
        else:
            end_frame = start_frame

        # Parse metadata (expects a JSON string like '{"1":"grab box"}')
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = ast.literal_eval(metadata_str)
        label = metadata.get("1", None)
        if label is None:
            print(f"Warning: No label found in metadata: {metadata_str}")
            continue

        # Check if the label is in the list of accepted labels
        if label not in accepted_labels:
            print(f"Skipping annotation with label '{label}' (not accepted).")
            continue

        # Look up the unique numeric label from the hard-coded mapping dictionary
        if label not in mapping_dict:
            print(f"Warning: Label '{label}' not found in mapping.")
            continue
        numeric_label = mapping_dict[label]

        output_rows.append([address, start_frame, end_frame, numeric_label])
    
    # Write the output CSV with header columns: address, start_frame, end_frame, numeric_label
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["address", "start_frame", "end_frame", "numeric_label"])
        writer.writerows(output_rows)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    # Set your file paths and subject identifier here:
    for i in range(1, 11):
        subject_key = f'S{i:02d}'   
        annotations_path = f"/Users/dennisbaumann/cars_paper/data/Arctic_Annotations/CSV/Arctic-{subject_key}.csv"    # Replace with your annotations CSV file path
                                # Set the subject identifier (e.g., S01, S02, etc.)
        output_path = f"/Users/dennisbaumann/cars_paper/data/Arctic_Annotations/CSV/better_annotations/{subject_key}.csv"               # Replace with the desired output CSV file path

        

        # Hard-coded mapping dictionary with unique numerical values:
        mapping_dict = {
            "nothing": 0,
            "close box": 1,
            "grab phone": 2,
            "grab waffleiron": 3,
            "rotate microwave": 4,
            "place mixer": 5,
            "place ketchup": 6,
            "open microwave": 7,
            "rotate waffleiron": 8,
            "grab scissors": 9,
            "open capsulemachine": 10,
            "close ketchup": 11,
            "open laptop": 12,
            "grab notebook": 13,
            "rotate ketchup": 14,
            "close mixer": 15,
            "open notebook": 16,
            "place microwave": 17,
            "rotate laptop": 18,
            "grab microwave": 19,
            "close waffleiron": 20,
            "grab mixer": 21,
            "place scissors": 22,
            "rotate phone": 23,
            "rotate espressomachine": 24,
            "use ketchup": 25,
            "place laptop": 26,
            "read notebook": 27,
            "close phone": 28,
            "open box": 29,
            "place box": 30,
            "grab capsulemachine": 31,
            "open phone": 32,
            "place notebook": 33,
            "grab ketchup": 34,
            "grab box": 35,
            "grab laptop": 36,
            "rotate mixer": 37,
            "lever espressomachine": 38,
            "open mixer": 39,
            "type phone": 40,
            "rotate capsulemachine": 41,
            "open waffleiron": 42,
            "close capsulemachine": 43,
            "close notebook": 44,
            "knob espressomachine": 45,
            "dial phone": 46,
            "rotate box": 47,
            "type laptop": 48,
            "close laptop": 49,
            "close microwave": 50,
            "open ketchup": 51,
            "place waffleiron": 52,
            "grab espressomachine": 53,
            "place phone": 54,
            "cut scissors": 55,
            "rotate notebook": 56,
            "place espressomachine": 57,
            "place capsulemachine": 58
        }
        accepted_labels = mapping_dict.keys()
        parse_annotations(annotations_path, mapping_dict, subject_key, output_path, accepted_labels)



