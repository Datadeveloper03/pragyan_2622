# Hybrid Clinical Parser 🩺

A Python-based utility designed to extract clinical vitals and chronic disease information from PDF medical records. It uses a **hybrid OCR strategy** to maximize accuracy while maintaining speed.

## 🚀 Features

- **Hybrid Extraction:** Automatically detects if a PDF page contains digital text (via `PyMuPDF`) or requires OCR (via `RapidOCR`).
- **Medical Intelligence:** Includes clinical sanity checks (e.g., filtering out impossible temperatures or SpO2 levels).
- **Automatic Unit Conversion:** Detects Fahrenheit and automatically converts to Celsius if the value exceeds 50.
- **Risk-Focused Logic:** When multiple values are found, it prioritizes "high-risk" markers (e.g., highest temperature, lowest oxygen saturation).
- **Chronic Disease Counter:** Scans for key comorbidities like Diabetes, COPD, and Hypertension.

---

## 🛠️ Architecture

The parser follows a tiered logic flow to ensure no data is missed regardless of how the PDF was generated.

1. **Text Check:** If a page contains $>50$ characters of digital text, it bypasses OCR for speed.
2. **OCR Fallback:** For scans, it renders the page at **2.0x zoom** to ensure medical jargon and small decimals are legible for the engine.
3. **Regex Parsing:** Uses specialized regular expressions to find Vitals, Age, and Medical History.

---

## 📦 Installation

Ensure you have the following dependencies installed:

Bash

# 

`pip install pymupdf rapidocr_onnxruntime`

*Note: `rapidocr_onnxruntime` is used for its balance of high accuracy and low CPU overhead compared to Tesseract.*

---

## 💻 Usage

The class is designed to work seamlessly with **Streamlit** `file_uploader` or standard file buffers.

Python

# 

`from your_script_name import HybridClinicalParser

# Initialize the engine
parser = HybridClinicalParser()

# Pass a file-like object (e.g., from Streamlit or open())
with open("patient_report.pdf", "rb") as f:
    vitals_data = parser.extract_from_file(f)

# Example Output
print(vitals_data)
# {
#   'body_temperature': 38.2,
#   'oxygen_saturation': 94,
#   'heart_rate': 110,
#   'systolic_blood_pressure': 145,
#   'age': 67,
#   'chronic_disease_count': 2,
#   'raw_text': "..."
# }`

---

## 🔍 Extraction Logic Table

| **Metric** | **Pattern Examples** | **Sanity Range** | **Strategy** |
| --- | --- | --- | --- |
| **Temperature** | `Temp: 38.2`, `101°F` | 30.0°C - 45.0°C | Max value (F to C conversion) |
| **SpO2** | `O2 Sat: 95%` | 50% - 100% | Min value (Risk detection) |
| **BP (Systolic)** | `BP: 120/80` | 50 - 250 mmHg | Max value |
| **Diseases** | `Hypertension`, `CAD` | N/A | Keyword count |
