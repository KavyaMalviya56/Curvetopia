import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN

def process_occlusion(paths):
    """Process occluded curves"""
    all_points = np.vstack(paths)
    
    # Use DBSCAN to cluster points
    clustering = DBSCAN(eps=10, min_samples=2).fit(all_points)
    labels = clustering.labels_
    
    completed_paths = []
    
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        
        cluster_points = all_points[labels == label]
        
        # Sort points to form a continuous path
        sorted_points = sort_points(cluster_points)
        
        # Complete the curve using cubic interpolation
        completed_curve = interpolate_curve(sorted_points)
        
        completed_paths.append(completed_curve)
    
    return completed_paths

def sort_points(points):
    """Sort points to form a continuous path"""
    distances = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    n = len(points)
    mask = np.ones(n, dtype=bool)
    ordered = [0]
    mask[0] = False
    
    for _ in range(n - 1):
        last = ordered[-1]
        next_idx = np.argmin(distances[last][mask])
        ordered.append(np.arange(n)[mask][next_idx])
        mask[ordered[-1]] = False
    
    return points[ordered]

def interpolate_curve(points, num_points=100):
    """Interpolate the curve using cubic interpolation"""
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    
    # Use cubic interpolation if we have enough points, otherwise use linear
    kind = 'cubic' if len(points) > 3 else 'linear'
    
    fx = interp1d(t, points[:, 0], kind=kind)
    fy = interp1d(t, points[:, 1], kind=kind)
    
    return np.column_stack([fx(t_new), fy(t_new)])