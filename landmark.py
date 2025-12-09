import numpy as np
import cv2
from skimage import measure

def get_edge_landmarks(img, n_points=20, threshold=0.15):
    """
    Extract n evenly-spaced points from the tissue edge.
    img: 2D numpy array (normalized image)
    n_points: number of landmarks
    threshold: used to create a binary mask (tune for your dataset)
    """

    # 1. Binary mask of tissue
    mask = img > threshold * img.max()
    mask = mask.astype(np.uint8)

    # 2. Find contours (OpenCV expects uint8 image)
    contours, hierarchy = cv2.findContours(mask, 
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No tissue detected—adjust threshold.")

    # 3. Pick the largest contour (main tissue)
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]  # shape (K, 2)

    # 4. Compute arc-length parameterization
    arc = cv2.arcLength(contour, closed=True)

    # 5. Sample n_points evenly
    sampled_points = []
    for i in range(n_points):
        d = (i / n_points) * arc
        point = cv2.pointPolygonTest(contour, tuple(contour[0]), measureDist=False)
        sampled_points.append(cv2.pointPolygonTest(contour, tuple(contour[0]), False))

    # Actually use cv2.approxPolyDP-like resampling via interpolation
    # Build cumulative distances along contour
    dists = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # Interpolate n evenly-spaced points
    target_distances = np.linspace(0, cumdist[-1], n_points)
    sampled = []
    for td in target_distances:
        idx = np.searchsorted(cumdist, td)
        idx = min(idx, len(contour)-1)
        sampled.append(contour[idx])

    sampled = np.array(sampled)
    return sampled
