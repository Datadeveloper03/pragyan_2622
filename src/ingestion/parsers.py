import pymupdf
from rapidocr_onnxruntime import RapidOCR
import tempfile
import os
import re

class BaseClinicalParser:
    """Holds the shared regex extraction logic so we don't duplicate code."""
    def _parse_vitals(self, text):
        vitals = {}
        
        # 1. Temperature: Explicitly handles filler words like "temp to 104.4" or "temp of"
        t_matches = re.findall(r'(?:Temperature|Temp)\s*(?:of|to|is|at|:)?\s*(\d{2,3}\.?\d?)', text, re.I)
        t_deg_matches = re.findall(r'(\d{2,3}\.?\d?)\s*°\s*[FfCc]', text)
        
        valid_temps = [round((val - 32) * 5 / 9, 1) if val > 50 else val for val in [float(t) for t in t_matches + t_deg_matches]]
        valid_temps_c = [t for t in valid_temps if 30.0 <= t <= 45.0]
        if valid_temps_c: 
            vitals['body_temperature'] = max(valid_temps_c)
            
        # 2. SpO2: Handles the zero/letter-O confusion "[Oo0]2" and filler words
        spo2_matches = re.findall(r'(?:SpO2|Oxygen\s*Saturation|[Oo0]2\s*[Ss]at)\s*(?:of|to|is|at|:)?\s*(\d{2,3})', text, re.I)
        valid_spo2 = [int(s) for s in spo2_matches if 50 <= int(s) <= 100]
        if valid_spo2: 
            vitals['oxygen_saturation'] = min(valid_spo2)
            
        # 3. Heart Rate: Catches "HR of 110" or "Pulse: 110"
        hr_matches = re.findall(r'(?:Heart\s*Rate|HR|Pulse)\s*(?:of|to|is|at|:)?\s*(\d{2,3})', text, re.I)
        valid_hr = [int(h) for h in hr_matches if 30 <= int(h) <= 250]
        if valid_hr: 
            vitals['heart_rate'] = max(valid_hr)
            
        # 4. Blood Pressure: Added leniency for spaces and "of/is"
        sbp_matches = re.findall(r'(?:Blood\s*Pressure|BP)\s*(?:of|to|is|at|:)?\s*(\d{2,3})\s*/', text, re.I)
        valid_sbp = [int(b) for b in sbp_matches if 50 <= int(b) <= 300]
        if valid_sbp: 
            vitals['systolic_blood_pressure'] = max(valid_sbp)

        # 5. Age
        age_match = re.search(r'(?:Age\s*[:\n\t]+(\d{1,3}))|(\d{1,3})\s*[-]?\s*(?:year[s]?\s*[-]?\s*old|y\.?o\.?)', text, re.I)
        if age_match:
            age_val = int(age_match.group(1) or age_match.group(2))
            if 0 <= age_val <= 120: 
                vitals['age'] = age_val

        # 6. Chronic Disease Count
        chronic_keywords = ['hypertension', 'asthma', 'diabetes', 'psoriatic arthritis', 'coronary artery disease', 'cad', 'copd', 'cancer', 'heart failure']
        disease_count = sum(1 for kw in chronic_keywords if re.search(r'\b' + kw + r'\b', text, re.I))
        if disease_count > 0:
            vitals['chronic_disease_count'] = disease_count
        
        vitals['raw_text'] = text 
        return vitals

class DigitalPDFParser(BaseClinicalParser):
    """Ultra-fast parser strictly for native digital PDFs."""
    def extract_from_file(self, file_upload):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_upload.getvalue())
            tmp_path = tmp.name

        doc = pymupdf.open(tmp_path)
        full_text = " ".join([page.get_text().strip() for page in doc])
        doc.close()
        os.remove(tmp_path)
        
        return self._parse_vitals(full_text)

class OCRImageParser(BaseClinicalParser):
    """Deep visual parser for Scanned PDFs and Images."""
    def __init__(self):
        self.engine = RapidOCR() # Only loads when specifically requested

    def extract_from_file(self, file_upload):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_upload.getvalue())
            tmp_path = tmp.name

        doc = pymupdf.open(tmp_path)
        full_text = []

        for page in doc:
            mat = pymupdf.Matrix(2.0, 2.0) 
            pix = page.get_pixmap(matrix=mat)
            result, _ = self.engine(pix.tobytes("png"))
            if result:
                result.sort(key=lambda x: x[0][0][1])
                full_text.append("\n".join([line[1] for line in result]))

        doc.close()
        os.remove(tmp_path)
        return self._parse_vitals("\n\n".join(full_text))