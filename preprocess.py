import cv2
import numpy as np


# ── CNN input preprocessing ───────────────────────────────────
def preprocess_image(img):
    """
    Convert BGR image → grayscale → resize 128×128 → normalise →
    reshape to (1, 128, 128, 1) for CNN input.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    gray = gray.astype("float32") / 255.0
    return gray.reshape(1, 128, 128, 1)


# ── OCR enhancement ───────────────────────────────────────────
def _sharpness(gray):
    """Laplacian variance — higher = sharper = better OCR orientation."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance_for_ocr(img):
    """
    1. Try 0°/90°/180°/270° — pick sharpest orientation.
    2. Upscale to at least 1200px wide (Tesseract loves high-res).
    3. CLAHE for adaptive contrast.
    4. Gaussian denoise.
    5. Otsu binarization.
    Returns a clean grayscale binary image ready for Tesseract.
    """
    best_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_score = _sharpness(best_gray)

    for angle in [90, 180, 270]:
        h, w   = img.shape[:2]
        M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot    = cv2.warpAffine(img, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        g      = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        score  = _sharpness(g)
        if score > best_score:
            best_score = score
            best_gray  = g

    # Upscale if too small — Tesseract performs badly below ~1000px wide
    h, w = best_gray.shape
    if w < 1200:
        scale     = 1200 / w
        best_gray = cv2.resize(best_gray, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # CLAHE
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(best_gray)

    # Gaussian blur to reduce noise before binarization
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Otsu binarization
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ── Image quality stats ───────────────────────────────────────
def get_image_stats(img):
    """Return brightness, contrast and sharpness for the quality panel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(np.mean(gray)),
        "contrast":   float(np.std(gray)),
        "sharpness":  float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
