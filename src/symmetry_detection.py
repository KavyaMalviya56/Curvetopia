import numpy as np
from scipy.spatial.distance import cdist

def detect_symmetry(points, threshold=0.05):
    points = np.array(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    
    def reflection_error(axis):
        reflected = points.copy()
        reflected[:, axis] = 2 * center[axis] - reflected[:, axis]
        distances = cdist(points, reflected)
        return np.mean(np.min(distances, axis=1))
    
    # Vertical symmetry
    vertical_error = reflection_error(0)
    
    # Horizontal symmetry
    horizontal_error = reflection_error(1)
    
    # Rotational symmetry
    def rotational_error(n):
        angle = 2 * np.pi / n
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated = np.dot(points - center, rotation_matrix.T) + center
        distances = cdist(points, rotated)
        return np.mean(np.min(distances, axis=1))
    
    rotational_errors = [rotational_error(n) for n in range(2, 9)]
    best_rotational = np.argmin(rotational_errors) + 2
    
    min_error = min(vertical_error, horizontal_error, min(rotational_errors))
    scale = np.ptp(points, axis=0).mean()
    
    if min_error < threshold * scale:
        if min_error == vertical_error:
            return "vertical", center[0]
        elif min_error == horizontal_error:
            return "horizontal", center[1]
        else:
            return f"rotational-{best_rotational}", center
    
    return "none", None