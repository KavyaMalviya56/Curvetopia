import os
import xml.etree.ElementTree as ET
import numpy as np
import traceback
from src.data_processing import read_csv
from src.shape_detection import regularize_curve
from src.symmetry_detection import detect_symmetry
from src.curve_completion import complete_curve
from src.curve_processing import process_occlusion

def process_curves(paths, filename):
    if filename.startswith("occlusion"):
        completed_curves = process_occlusion(paths)
        return [{"shape_type": "completed", "completed_curve": curve} for curve in completed_curves]
    else:
        regularized_paths = []
        for i, path in enumerate(paths):
            try:
                shape_type, shape_params = regularize_curve(path)
                symmetry_type, symmetry_axis = detect_symmetry(path)
                completed_curve = complete_curve(shape_params, shape_type)
                regularized_paths.append({
                    "shape_type": shape_type,
                    "completed_curve": completed_curve,
                    "symmetry_type": symmetry_type,
                    "symmetry_axis": symmetry_axis,
                    "shape_params": shape_params
                })
            except Exception as e:
                print(f"Error processing curve {i}:")
                print(traceback.format_exc())
                regularized_paths.append({
                    "shape_type": "unknown",
                    "completed_curve": path,
                    "symmetry_type": "none",
                    "symmetry_axis": None,
                    "shape_params": None
                })
        return regularized_paths

def create_svg(regularized_paths, output_file):
    svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'version': '1.1'})
    
    # Calculate bounding box
    all_points = np.concatenate([path['completed_curve'] for path in regularized_paths if path['completed_curve'] is not None])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    
    # Add some padding
    padding = 0.05 * max(max_x - min_x, max_y - min_y)
    min_x, min_y = min_x - padding, min_y - padding
    max_x, max_y = max_x + padding, max_y + padding
    
    svg.set('viewBox', f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}")
    
    for path in regularized_paths:
        shape_type = path['shape_type']
        points = path['completed_curve']
        
        if points is None or len(points) < 2:
            continue
        
        # Filter out points outside the bounding box
        mask = ((points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
                (points[:, 1] >= min_y) & (points[:, 1] <= max_y))
        points = points[mask]
        
        if len(points) < 2:  # Skip if not enough points after filtering
            continue
        
        if shape_type in ["unknown", "completed"]:
            # Use cubic Bezier curves for smoother rendering
            path_data = f"M {points[0][0]},{points[0][1]}"
            for i in range(1, len(points) - 2, 3):
                path_data += f" C {points[i][0]},{points[i][1]} {points[i+1][0]},{points[i+1][1]} {points[i+2][0]},{points[i+2][1]}"
            ET.SubElement(svg, 'path', {
                'd': path_data,
                'fill': 'none', 'stroke': 'yellow', 'stroke-width': '2'
            })
        elif shape_type == "line":
            ET.SubElement(svg, 'line', {
                'x1': str(points[0, 0]), 'y1': str(points[0, 1]),
                'x2': str(points[-1, 0]), 'y2': str(points[-1, 1]),
                'stroke': 'yellow', 'stroke-width': '2'
            })
        elif shape_type == "circle":
            center, radius = path['shape_params']
            ET.SubElement(svg, 'circle', {
                'cx': str(center[0]), 'cy': str(center[1]), 'r': str(radius),
                'fill': 'none', 'stroke': 'yellow', 'stroke-width': '2'
            })
        elif shape_type == "ellipse":
            center, axes, angle = path['shape_params']
            ET.SubElement(svg, 'ellipse', {
                'cx': str(center[0]), 'cy': str(center[1]),
                'rx': str(axes[0]/2), 'ry': str(axes[1]/2),
                'transform': f'rotate({angle},{center[0]},{center[1]})',
                'fill': 'none', 'stroke': 'yellow', 'stroke-width': '2'
            })
        else:
            points_str = ' '.join(f"{x},{y}" for x, y in points)
            ET.SubElement(svg, 'polygon', {
                'points': points_str,
                'fill': 'none', 'stroke': 'yellow', 'stroke-width': '2'
            })
        
        if 'symmetry_type' in path and path['symmetry_type'] != "none":
            if path['symmetry_type'] == "vertical":
                x = path['symmetry_axis']
                ET.SubElement(svg, 'line', {
                    'x1': str(x), 'y1': str(min_y),
                    'x2': str(x), 'y2': str(max_y),
                    'stroke': 'red', 'stroke-dasharray': '5,5', 'stroke-width': '1'
                })
            elif path['symmetry_type'] == "horizontal":
                y = path['symmetry_axis']
                ET.SubElement(svg, 'line', {
                    'x1': str(min_x), 'y1': str(y),
                    'x2': str(max_x), 'y2': str(y),
                    'stroke': 'red', 'stroke-dasharray': '5,5', 'stroke-width': '1'
                })

    tree = ET.ElementTree(svg)
    tree.write(output_file)

def process_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.svg")
            
            try:
                paths = read_csv(input_path)
                regularized_paths = process_curves(paths, filename)
                create_svg(regularized_paths, output_path)
                print(f"Processed {filename}. Output saved to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}:")
                print(traceback.format_exc())

def main(input_directory, output_directory):
    try:
        process_directory(input_directory, output_directory)
        print("All files processed.")
    except Exception as e:
        print(f"An error occurred during processing:")
        print(traceback.format_exc())

if __name__ == "__main__":
    input_directory = "data/input/"  # Replace with actual input directory path
    output_directory = "data/"  # Replace with desired output directory path
    main(input_directory, output_directory)