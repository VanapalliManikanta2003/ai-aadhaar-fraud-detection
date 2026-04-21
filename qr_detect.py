import cv2

def detect_qr(image):
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(image)
    if bbox is not None and data:
        return True, data
    else:
        return False, None
