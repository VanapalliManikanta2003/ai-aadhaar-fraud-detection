import pytesseract
import re
import numpy as np
import cv2

# ── Tesseract path (Windows) ─────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Multi-pass PSM configs for best OCR on ID cards ─────────
PSM_CONFIGS = [
    "--oem 3 --psm 6",   # Uniform block of text
    "--oem 3 --psm 4",   # Single column
    "--oem 3 --psm 3",   # Fully automatic
    "--oem 3 --psm 11",  # Sparse text
]

def extract_text(img):
    """
    Multi-pass Tesseract OCR.
    Tries 4 PSM configs and returns the longest (best) result.
    img: grayscale / binarized numpy array
    """
    best_text = ""
    best_len  = 0
    for cfg in PSM_CONFIGS:
        try:
            t = pytesseract.image_to_string(img, lang="eng", config=cfg)
            if len(t.strip()) > best_len:
                best_len  = len(t.strip())
                best_text = t
        except Exception:
            pass
    return best_text


def extract_fields(text):
    """
    Extract Aadhaar Number, DOB, Name, Gender from raw OCR text.

    Aadhaar rules:
      - 12 digits grouped as XXXX XXXX XXXX or XXXXXXXXXXXX
      - First digit is never 0 or 1 (UIDAI spec)

    DOB rules:
      - DD/MM/YYYY  or  DD-MM-YYYY
      - Also handles "DOB:" / "Year of Birth:" labels

    Name rules:
      - Printed in ALL CAPS on Aadhaar; line with 2-4 all-cap words
      - Avoid lines that are clearly not names (numbers, keywords)

    Gender rules:
      - "MALE" / "FEMALE" appear in caps on the card
    """

    # ── Aadhaar Number ────────────────────────────────────────
    # Match  XXXX XXXX XXXX  or  XXXX-XXXX-XXXX  or  12 digits together
    aadhaar_patterns = [
        r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b",   # spaced / hyphenated
        r"\b[2-9]\d{11}\b",                           # no separator
    ]
    aadhaar = None
    for pat in aadhaar_patterns:
        m = re.findall(pat, text)
        if m:
            # Normalise to XXXX XXXX XXXX
            raw = re.sub(r"[\s\-]", "", m[0])
            aadhaar = f"{raw[:4]} {raw[4:8]} {raw[8:]}"
            break

    # ── Date of Birth ─────────────────────────────────────────
    dob = None
    # Handle "DOB: 01/01/1990" labels first
    dob_label = re.search(
        r"(?:DOB|Date of Birth|D\.O\.B)[:\s]+(\d{2}[\/\-]\d{2}[\/\-]\d{4})",
        text, re.IGNORECASE
    )
    if dob_label:
        dob = dob_label.group(1)
    else:
        dob_plain = re.findall(r"\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b", text)
        if dob_plain:
            dob = dob_plain[0]
    # Also handle "Year of Birth: 1990"
    if not dob:
        yob = re.search(r"(?:Year of Birth|YOB)[:\s]+(\d{4})", text, re.IGNORECASE)
        if yob:
            dob = yob.group(1)

    # ── Name ──────────────────────────────────────────────────
    # Aadhaar prints name as ALL CAPS line e.g. "VANAPALLI MANIKANTA"
    name = None
    SKIP_WORDS = {
        "GOVERNMENT", "INDIA", "UIDAI", "UNIQUE", "IDENTIFICATION",
        "AUTHORITY", "AADHAAR", "ADDRESS", "MALE", "FEMALE",
        "DOB", "DATE", "BIRTH", "VID", "HELP", "TOLL", "FREE",
        "RESIDENT", "ENROLLMENT",
    }
    for line in text.split("\n"):
        line = line.strip()
        # Must be 2-5 words, all alphabetic uppercase tokens
        tokens = line.split()
        if 2 <= len(tokens) <= 5:
            if all(re.match(r"^[A-Z]+$", tok) for tok in tokens):
                if not any(tok in SKIP_WORDS for tok in tokens):
                    name = " ".join(tokens)
                    break

    # Fallback: title-case name (e.g. "Vanapalli Manikanta")
    if not name:
        tc = re.findall(r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})", text)
        filtered = [
            n for n in tc
            if not any(kw in n.upper() for kw in SKIP_WORDS)
        ]
        if filtered:
            name = filtered[0].strip()

    # ── Gender ────────────────────────────────────────────────
    gender = None
    tl = text.upper()
    # Check word boundaries to avoid partial matches
    if re.search(r"\bFEMALE\b", tl):
        gender = "Female"
    elif re.search(r"\bMALE\b", tl):
        gender = "Male"

    return {
        "Aadhaar": aadhaar  if aadhaar  else "Not Found",
        "DOB":     dob      if dob      else "Not Found",
        "Name":    name     if name     else "Not Found",
        "Gender":  gender   if gender   else "Not Found",
    }


def get_ocr_confidence(img):
    """Return mean Tesseract confidence score 0-100."""
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6"
        )
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        return float(np.mean(confs)) if confs else 0.0
    except Exception:
        return 0.0
