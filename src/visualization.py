import matplotlib.pyplot as plt
import numpy as np

def plot(paths_XYs, regularized_paths=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colours = plt.cm.rainbow(np.linspace(0, 1, len(paths_XYs)))
    
    # Plot original paths
    for i, XYs in enumerate(paths_XYs):
        c = colours[i]
        for XY in XYs:
            ax1.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax1.set_title("Original Paths")
    ax1.set_aspect('equal')
    
    # Plot regularized paths if provided
    if regularized_paths:
        for i, path in enumerate(regularized_paths):
            c = colours[i]
            for curve in path:
                shape = curve['shape']
                params = curve['params']
                if shape == 'line':
                    vx, vy, x, y = params
                    ax2.plot([x-vx*100, x+vx*100], [y-vy*100, y+vy*100], c=c)
                elif shape == 'circle':
                    xc, yc, r = params
                    circle = plt.Circle((xc, yc), r, fill=False, color=c)
                    ax2.add_artist(circle)
                elif shape == 'ellipse':
                    center, axes, angle = params
                    ellipse = plt.patches.Ellipse(center, axes[0], axes[1], angle, fill=False, color=c)
                    ax2.add_artist(ellipse)
                elif shape in ['rectangle', 'polygon']:
                    ax2.plot(params[:, 0], params[:, 1], c=c)
                elif shape == 'star':
                    centroid, peaks = params
                    ax2.plot(points[peaks, 0], points[peaks, 1], c=c, marker='*')
                else:
                    ax2.plot(params[:, 0], params[:, 1], c=c)
        ax2.set_title("Regularized Paths")
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()