import numpy as np
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

    # 1. Binary mask
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
    sampled = []
    for td in target_distances:
        idx = np.searchsorted(cumdist, td)
        idx = min(idx, len(contour)-1)
        sampled.append(contour[idx])

    return np.array(sampled)
