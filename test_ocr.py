import sys
import cv2
from ocr.ocr_extract import extract_text, extract_fields, get_ocr_confidence
from preprocessing.preprocess import enhance_for_ocr

img_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_all/genuine/real_processed_1.jpg"
img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found:", img_path)
    exit()

enhanced = enhance_for_ocr(img)
text     = extract_text(enhanced)
fields   = extract_fields(text)
conf     = get_ocr_confidence(enhanced)

print("OCR TEXT:\n", text)
print("FIELDS:",  fields)
print(f"OCR Confidence: {conf:.1f}%")
