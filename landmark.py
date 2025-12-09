import numpy as np
import matplotlib.pyplot as plt 
import cv2

def get_edge_landmarks(Xg, n_points=20, threshold=0.01):
    """
    Extract n evenly-spaced points from the tissue edge.
    Xg: 3D tensor array of (n_genes, n_rows, n_cols)
    n_points: number of landmarks
    threshold: used to create a binary mask (tune for your dataset)
    """
    print(f"Selecting {n_points} landmark points at threshold {threshold}...")

    # 0. Normalize the image 
    img = np.mean(Xg, axis=0)

    # 1. Binary mask of tissue
    mask = (img > threshold * img.max()).astype(np.uint8)

    # 2. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No tissue detected—adjust threshold.")

    # 3. Pick the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]  # shape (K, 2)

    # 4. Compute cumulative distances along contour
    diffs = np.diff(contour, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # 5. Interpolate n evenly-spaced points
    target_distances = np.linspace(0, cumdist[-1], n_points)
    sampled = []
    for td in target_distances:
        idx = np.searchsorted(cumdist, td)
        idx = min(idx, len(contour)-1)
        sampled.append(contour[idx])

    sampled = np.array(sampled)
    print("Returning sampled points...")
    return sampled

def visualize_points(XgI, XgJ, pointsI, pointsJ): 
    """
    Display source and target images with landmarks.

    Parameters
    ----------
    XgI, XgJ : np.ndarray (a 3D tensor)
        Rasterized images (n_genes, nrows, ncols)
    pointsI, pointsJ : np.ndarray
        Landmark coordinates in (row, col) format
    """
    # pointsI and pointsJ are the landmarks from get_edge_landmarks
    imgI = np.mean(XgI, axis=0)
    imgJ = np.mean(XgJ, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for XgI
    axs[0].imshow(imgI, cmap='gray')
    axs[0].scatter(pointsI[:, 0], pointsI[:, 1], c='red', s=40)
    axs[0].set_title('Landmarks on XgI')
    axs[0].axis('off')

    # Plot for XgJ
    axs[1].imshow(imgJ, cmap='gray')
    axs[1].scatter(pointsJ[:, 0], pointsJ[:, 1], c='red', s=40)
    axs[1].set_title('Landmarks on XgJ')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()