import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN

def complete_occluded_curve(paths):
    # Combine all paths into a single array of points
    all_points = np.vstack(paths)
    
    if len(all_points) < 2:
        return paths  # Not enough points to process
    
    # Use DBSCAN to cluster points and identify gaps
    clustering = DBSCAN(eps=10, min_samples=2).fit(all_points)
    labels = clustering.labels_
    
    completed_paths = []
    
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        
        cluster_points = all_points[labels == label]
        
        if len(cluster_points) < 2:
            continue  # Not enough points in this cluster
        
        # Sort points to form a continuous path
        sorted_points = sort_points(cluster_points)
        
        # Identify gaps in the sorted points
        gaps = find_gaps(sorted_points)
        
        # Complete the curve by interpolating through the gaps
        completed_curve = interpolate_gaps(sorted_points, gaps)
        
        # Refine the completed curve using a neural network
        refined_curve = refine_curve_with_nn(completed_curve)
        
        completed_paths.append(refined_curve)
    
    return completed_paths if completed_paths else paths

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
        
        # Use linear interpolation if spline fails
        try:
            tck, u = splprep(segment.T, s=0, k=min(3, len(segment) - 1))
            u_new = np.linspace(0, 1, num=100)
            interpolated = np.column_stack(splev(u_new, tck))
        except ValueError:
            t = np.linspace(0, 1, len(segment))
            t_new = np.linspace(0, 1, 100)
            interpolated = np.column_stack([np.interp(t_new, t, segment[:, i]) for i in range(2)])
        
        completed_curve.extend(interpolated)
    
    return np.array(completed_curve)

def refine_curve_with_nn(curve):
    if len(curve) < 2:
        return curve
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(curve)
    
    # Create and train the model
    model = MLPRegressor(hidden_layer_sizes=(64, 32, 64),
                         max_iter=1000,
                         activation='relu',
                         solver='adam',
                         random_state=42)
    
    # Add noise to input for robustness
    noisy_input = normalized_points + np.random.normal(0, 0.01, normalized_points.shape)
    
    model.fit(noisy_input, normalized_points)
    
    # Use the model to refine points
    refined_points = model.predict(normalized_points)
    
    # Denormalize the refined points
    refined_points = scaler.inverse_transform(refined_points)
    
    return refined_points

def process_occlusion(paths):
    return complete_occluded_curve(paths)