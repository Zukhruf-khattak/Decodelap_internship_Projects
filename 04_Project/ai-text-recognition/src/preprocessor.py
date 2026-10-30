"""Phase 2: PROCESS — Image Pre-processing.

Raw visual data is cluttered. We use OpenCV to:
1. Convert to Grayscale (collapse 3D RGB to 1D intensity).
2. Gaussian Blur (remove micro-imperfections).
3. Adaptive Thresholding (force binary black/white decision).
"""

import cv2

def preprocess_image(image_path: str, output_path: str = None) -> any:
    """Read, grayscale, blur, and threshold an image.
    
    Returns the final processed image matrix.
    """
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian Blur (kernel size 5x5, standard deviation 0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Adaptive Thresholding (Otsu's Method)
    # This forces the image into pure black and white, separating foreground from noise.
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU automatically calculates the optimal threshold.
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Save the processed image for visual confirmation
    if output_path:
        cv2.imwrite(output_path, thresh)

    return thresh