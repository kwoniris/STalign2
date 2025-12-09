import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_edge_landmarks(img, n_points=20, threshold=0.15):
    """
    Extract n evenly-spaced points from the tissue edge.

    Parameters:
    -----------
    img : np.ndarray
        2D normalized image (float32 or float64)
    n_points : int
        Number of landmarks to extract
    threshold : float
        Fraction of max intensity to define tissue mask

    Returns:
    --------
    landmarks : np.ndarray
        Array of shape (n_points, 2) with (x, y) coordinates
    """

    # 1. Convert image to a binary mask 
    mask = img > threshold * img.max()
    mask = mask.astype(np.uint8)

    # 2. Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No tissue detected—adjust threshold.")

    # 3. Pick the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]  # reshape (N,2)

    # 4. Compute cumulative distances along contour
    dists = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # 5. Sample n_points evenly spaced points along the contour
    target_distances = np.linspace(0, cumdist[-1], n_points)
    # print(target_distances)
    # differences = [target_distances[i+1]-target_distances[i] for i in range(len(target_distances)-1)]
    # print("Differences", differences)
    sampled = []
    for td in target_distances:
        idx = np.searchsorted(cumdist, td)
        idx = min(idx, len(contour)-1)
        sampled.append(contour[idx])

    return np.array(sampled)


def visualize_points(XgI, XgJ, pointsI, pointsJ): 
    # Suppose your images are 2D arrays
    img_source = np.mean(XgI, axis=0)  # average across genes if you want
    img_target = np.mean(XgJ, axis=0)

    # pointsI and pointsJ should be in (row, col) format
    plt.figure(figsize=(12,5))

    # Source image
    plt.subplot(1,2,1)
    plt.imshow(img_source, cmap='viridis')
    plt.scatter(pointsI[:,1], pointsI[:,0], c='r', s=40)  # col=x, row=y
    plt.title('Source Image with Landmarks')
    plt.axis('off')

    # Target image
    plt.subplot(1,2,2)
    plt.imshow(img_target, cmap='viridis')
    plt.scatter(pointsJ[:,1], pointsJ[:,0], c='b', s=40)
    plt.title('Target Image with Landmarks')
    plt.axis('off')

    plt.show()