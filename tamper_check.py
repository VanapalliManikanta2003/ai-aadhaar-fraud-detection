import cv2
import numpy as np

def tamper_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if np.mean(edges) < 5:
        return "⚠ Possible Tampering"
    return "✔ Looks Normal"
