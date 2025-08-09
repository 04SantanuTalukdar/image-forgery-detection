import cv2
import numpy as np

def extract_noise_residual(image):
    """
    Extract noise residual using a high-pass filter.

    image: numpy array (H x W x 3) in RGB format, uint8
    returns: numpy array (H x W), single channel noise residual normalized to [0,1]
    """

    # RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    noise_residual = cv2.filter2D(gray, -1, kernel)

    noise_residual = (noise_residual - noise_residual.min()) / (noise_residual.max() - noise_residual.min() + 1e-8)
    return noise_residual.astype(np.float32)
