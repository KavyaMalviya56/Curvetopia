import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def ensure_float_points(points):
    return np.array(points, dtype=np.float32)

def remove_outliers(points, eps=5, min_samples=2):
    if len(points) < min_samples:
        return points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return points[clustering.labels_ != -1]

def detect_line(points, epsilon=1e-3):
    if len(points) < 2:
        return False, None
    points = ensure_float_points(points)
    points = remove_outliers(points)
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    distances = np.abs(np.cross(points - [x[0], y[0]], [vx[0], vy[0]])) / np.linalg.norm([vx[0], vy[0]])
    return np.all(distances < epsilon * np.ptp(points)), (vx[0], vy[0], x[0], y[0])

def detect_circle(points, epsilon=1e-3):
    if len(points) < 3:
        return False, None
    points = ensure_float_points(points)
    points = remove_outliers(points)
    center, radius = cv2.minEnclosingCircle(points)
    
    distances = cdist([center], points).flatten()
    inliers = points[distances < radius * (1 + epsilon)]
    if len(inliers) < len(points) * 0.9:
        return False, None
    
    center = np.mean(inliers, axis=0)
    radius = np.mean(np.linalg.norm(inliers - center, axis=1))
    
    distances = np.abs(cdist([center], points).flatten() - radius)
    return np.all(distances < epsilon * radius), (center, radius)

def detect_ellipse(points, epsilon=1e-3):
    if len(points) < 5:
        return False, None
    points = ensure_float_points(points)
    points = remove_outliers(points)
    try:
        ellipse = cv2.fitEllipse(points)
        center, axes, angle = ellipse
        cos_angle, sin_angle = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        x_centered, y_centered = points[:, 0] - center[0], points[:, 1] - center[1]
        x_rotated = x_centered * cos_angle + y_centered * sin_angle
        y_rotated = -x_centered * sin_angle + y_centered * cos_angle
        distances = np.abs(((x_rotated / (axes[0] / 2)) ** 2 + (y_rotated / (axes[1] / 2)) ** 2) - 1)
        return np.all(distances < epsilon), ellipse
    except:
        return False, None

def detect_rectangle(points, epsilon=1e-3, angle_threshold=5):
    if len(points) < 4:
        return False, None
    points = ensure_float_points(points)
    points = remove_outliers(points)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    
    edges = np.roll(box, -1, axis=0) - box
    angles = np.abs(np.arctan2(edges[:, 1], edges[:, 0]) * 180 / np.pi)
    if not np.all(np.abs(angles - 90) < angle_threshold):
        return False, None
    
    distances = np.min([
        np.abs(np.cross(box[i] - box[j], points - box[j]) / np.linalg.norm(box[i] - box[j]))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]
    ], axis=0)
    return np.all(distances < epsilon * min(rect[1])), rect

def detect_polygon(points, min_sides=3, max_sides=8, epsilon=1e-3):
    if len(points) < min_sides:
        return False, None
    points = ensure_float_points(points)
    points = remove_outliers(points)
    peri = cv2.arcLength(points, True)
    for n_sides in range(min_sides, max_sides + 1):
        approx = cv2.approxPolyDP(points, epsilon * peri, True)
        if len(approx) == n_sides:
            return True, approx
    return False, None

def regularize_curve(points):
    if len(points) == 0:
        return "unknown", points
    points = remove_outliers(points)
    detectors = [
        (detect_rectangle, "rectangle"),
        (detect_circle, "circle"),
        (detect_line, "line"),
        (detect_ellipse, "ellipse"),
        (detect_polygon, "polygon")
    ]
    for detector, shape_name in detectors:
        is_shape, params = detector(points)
        if is_shape:
            return shape_name, params
    return "unknown", points