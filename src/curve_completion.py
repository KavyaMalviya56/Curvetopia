import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN

def complete_curve(params, shape_type):
    if shape_type == "line":
        return complete_line(params)
    elif shape_type == "circle":
        return complete_circle(params)
    elif shape_type == "ellipse":
        return complete_ellipse(params)
    elif shape_type == "rectangle":
        return complete_rectangle(params)
    elif shape_type == "polygon":
        return complete_polygon(params)
    else:
        return complete_unknown(params)

def complete_line(params):
    vx, vy, x, y = params
    t = np.linspace(-1000, 1000, 100)
    return np.column_stack((x + t * vx, y + t * vy))

def complete_circle(params):
    center, radius = params
    theta = np.linspace(0, 2 * np.pi, 100)
    return np.column_stack((
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta)
    ))

def complete_ellipse(params):
    center, axes, angle = params
    t = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + axes[0]/2 * np.cos(t) * np.cos(np.radians(angle)) - axes[1]/2 * np.sin(t) * np.sin(np.radians(angle))
    y = center[1] + axes[0]/2 * np.cos(t) * np.sin(np.radians(angle)) + axes[1]/2 * np.sin(t) * np.cos(np.radians(angle))
    return np.column_stack((x, y))

def complete_rectangle(params):
    center, size, angle = params
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * (size[0]/2, size[1]/2)
    rotation = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                         [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
    rotated = np.dot(corners, rotation.T)
    return rotated + center

def complete_polygon(params):
    return params.reshape(-1, 2)

def complete_unknown(points):
    tck, u = splprep(points.T, s=0, k=3)
    unew = np.linspace(0, 1, 100)
    return np.column_stack(splev(unew, tck))

def process_occlusion(paths):
    # Combine all paths into a single array of points
    all_points = np.vstack(paths)
    
    # Use DBSCAN to cluster points and identify gaps
    clustering = DBSCAN(eps=10, min_samples=2).fit(all_points)
    labels = clustering.labels_
    
    completed_paths = []
    
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        
        cluster_points = all_points[labels == label]
        
        # Sort points to form a continuous path
        sorted_points = sort_points(cluster_points)
        
        # Identify gaps in the sorted points
        gaps = find_gaps(sorted_points)
        
        # Complete the curve by interpolating through the gaps
        completed_curve = interpolate_gaps(sorted_points, gaps)
        
        completed_paths.append(completed_curve)
    
    return completed_paths

def sort_points(points):
    # Start with the leftmost point
    sorted_indices = [np.argmin(points[:, 0])]
    remaining_indices = set(range(len(points))) - set(sorted_indices)
    
    while remaining_indices:
        last_point = points[sorted_indices[-1]]
        distances = np.linalg.norm(points[list(remaining_indices)] - last_point, axis=1)
        nearest_index = list(remaining_indices)[np.argmin(distances)]
        sorted_indices.append(nearest_index)
        remaining_indices.remove(nearest_index)
    
    return points[sorted_indices]

def find_gaps(points, threshold=20):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.where(distances > threshold)[0]

def interpolate_gaps(points, gaps):
    if len(gaps) == 0:
        return points
    
    # Add endpoints to the gaps
    gaps = np.concatenate(([0], gaps, [len(points)-1]))
    
    completed_curve = []
    
    for i in range(len(gaps) - 1):
        start, end = gaps[i], gaps[i+1]
        segment = points[start:end+1]
        
        if len(segment) < 2:
            continue
        
        # Use spline interpolation to smooth the segment and add points
        tck, u = splprep(segment.T, s=0, k=3)
        u_new = np.linspace(0, 1, num=100)
        interpolated = np.column_stack(splev(u_new, tck))
        
        completed_curve.extend(interpolated)
    
    return np.array(completed_curve)