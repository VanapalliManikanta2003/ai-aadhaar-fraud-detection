# AI-Powered Identity Verification and Fraud Prevention System using UID Aadhaar

## Project Structure
```
aadhar_fraud_detection/
├── app.py                        # Main Streamlit app
├── dataset.py                    # Dataset split script
├── check_images.py               # Check dataset image counts
├── train_model.py                # Train the CNN model
├── test_model.py                 # Evaluate model on test set
├── test_ocr.py                   # Test OCR extraction
├── requirements.txt              # Python dependencies
├── fraud_detection/
│   ├── cnn_model.py              # CNN architecture
│   └── tamper_check.py           # Tamper detection
├── preprocessing/
│   └── preprocess.py             # Image preprocessing
├── ocr/
│   └── ocr_extract.py            # OCR text extraction
├── qr/
│   └── qr_detect.py              # QR code detection
├── dataset_all/
│   ├── genuine/                  # Place genuine Aadhaar images here
│   └── fraud/                    # Place fraud Aadhaar images here
├── dataset/
│   ├── train/genuine/ & train/fraud/
│   ├── val/genuine/   & val/fraud/
│   └── test/genuine/  & test/fraud/
└── model/
    └── document_fraud_model.weights.h5  # Saved after training
```

## ⚙️ Prerequisites
- Python 3.8+
- Tesseract OCR installed: https://github.com/UB-Mannheim/tesseract/wiki

## 🚀 Run Steps (in order)

### Step 1: Install dependencies
```
pip install -r requirements.txt
```

### Step 2: Add images to dataset_all
Place your genuine images in `dataset_all/genuine/`
Place your fraud images in `dataset_all/fraud/`

### Step 3: Check image counts
```
python check_images.py
```

### Step 4: Split dataset
```
python dataset.py
```

### Step 5: Train the model
```
python train_model.py
```

### Step 6: Test the model
```
python test_model.py
```

### Step 7: Run the Streamlit app
```
streamlit run app.py
```

## Notes
- After training, model weights are saved at `model/document_fraud_model.weights.h5`
- Tesseract OCR path in `ocr/ocr_extract.py` may need to be updated for your system
