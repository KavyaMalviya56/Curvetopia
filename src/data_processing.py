import numpy as np
import csv

def read_csv(csv_path):
    paths = []
    current_path = []
    
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        current_path_id = None
        for row in csv_reader:
            path_id, subpath_id, x, y = map(float, row)
            
            if current_path_id is None or path_id != current_path_id:
                if current_path:
                    paths.append(np.array(current_path))
                current_path = []
                current_path_id = path_id
            
            current_path.append([x, y])
    
    if current_path:
        paths.append(np.array(current_path))
    
    return paths