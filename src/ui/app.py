import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.parsers import DigitalPDFParser, OCRImageParser
from src.explainability.explain import process_triage_and_shap
from src.explainability.medgemma import BioMistralExplainer

st.set_page_config(page_title="AI Clinical Triage Engine", page_icon="🏥", layout="wide")

# ==========================================
# 1. Initialize Global State (RAM Database)
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = {}
if 'triage_queue' not in st.session_state:
    st.session_state.triage_queue = [] 

@st.cache_resource
def load_llm():
    return BioMistralExplainer()

biomistral = load_llm()

st.sidebar.header("⚙️ Model Configuration")
all_available_features = [
    'age', 'heart_rate', 'systolic_blood_pressure', 'oxygen_saturation', 
    'body_temperature', 'pain_level', 'chronic_disease_count', 
    'previous_er_visits', 'arrival_mode'
]
selected_features = st.sidebar.multiselect(
    "Active Risk Features", 
    options=all_available_features, 
    default=['age', 'body_temperature', 'oxygen_saturation', 'heart_rate', 'pain_level', 'chronic_disease_count']
)

st.title("🏥 Enterprise AI Clinical Triage System")

# ==========================================
# 2. UI Layout: Enterprise Tabs
# ==========================================
tab1, tab2, tab3 = st.tabs(["📄 Document Ingestion", "✍️ Manual Intake", "🚨 Live Triage Board"])

# ------------------------------------------
# TAB 1: BATCH INGESTION
# ------------------------------------------
with tab1:
    st.header("Upload Clinical Records")
    processing_mode = st.radio(
        "Select Extraction Engine:", 
        ["⚡ Fast Digital Extraction (Native PDFs)", "🔍 Deep Vision OCR (Scans/Images)"],
        horizontal=True
    )
    
    files = st.file_uploader("Upload Scanned or Digital PDFs", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    
    if files:
        st.subheader("Map Patient IDs to Files")
        patient_map = {}
        cols = st.columns(min(len(files), 4))
        for i, file in enumerate(files):
            with cols[i % 4]:
                patient_map[file.name] = st.text_input(f"ID for {file.name[:15]}...", f"P-10{i+1}")

        if st.button("🚀 Run Batch Triage", type="primary"):
            if "Fast Digital" in processing_mode:
                parser = DigitalPDFParser()
            else:
                with st.spinner("Loading Deep Vision Engine..."):
                    parser = OCRImageParser()
                    
            for file in files:
                p_id = patient_map[file.name]
                with st.spinner(f"Analyzing {file.name}..."):
                    features = parser.extract_from_file(file)
                    features['pain_level'] = features.get('pain_level', 5)
                    features['arrival_mode'] = features.get('arrival_mode', 'walk_in')
                    features['chronic_disease_count'] = features.get('chronic_disease_count', 0)
                    features['previous_er_visits'] = features.get('previous_er_visits', 0)
                    
                    # 1. Get ML Risk Level
                    ai_result = process_triage_and_shap(features, selected_features)
                    triage_level = ai_result.get('triage_level', 0)
                    top_factors = ai_result.get('top_factors', [])
                    shap_str = ", ".join([f"{f['feature']} ({f['direction']})" for f in top_factors])
                    
                    # 2. Get all 3 outputs from the LLM using the Delimiter Trick
                    with st.spinner(f"Generating LLM Routing & Synthesis for {p_id}..."):
                        bio_dict = biomistral.get_explanation(triage_level, shap_str, features.get('raw_text', 'No clinical context.'))
                    
                    # 3. Longitudinal History
                    history_list = st.session_state.history.get(p_id, [])
                    trend = "stable"
                    deltas = {"spo2": None, "temp": None, "hr": None}
                    
                    if len(history_list) > 0:
                        prev = history_list[-1]['features']
                        if 'oxygen_saturation' in features and 'oxygen_saturation' in prev:
                            deltas['spo2'] = features['oxygen_saturation'] - prev['oxygen_saturation']
                        if 'body_temperature' in features and 'body_temperature' in prev:
                            deltas['temp'] = round(features['body_temperature'] - prev['body_temperature'], 1)
                        if 'heart_rate' in features and 'heart_rate' in prev:
                            deltas['hr'] = features['heart_rate'] - prev['heart_rate']

                        if features.get('oxygen_saturation', 100) < prev.get('oxygen_saturation', 100) or features.get('body_temperature', 37) > prev.get('body_temperature', 37):
                            trend = "worsening"

                    st.session_state.history.setdefault(p_id, []).append({"features": features, "triage_level": triage_level})
                    
                    # Map the Delimiter Dictionary out to the queue entry!
                    queue_entry = {
                        "id": p_id, "level": triage_level, "trend": trend, 
                        "features": features, "shap": shap_str, "deltas": deltas,
                        "bio_synthesis": bio_dict.get('short_synthesis', 'N/A'),
                        "bio_action": bio_dict.get('recommended_action', 'N/A'),
                        "bio_department": bio_dict.get('department_routing', 'N/A')
                    }
                    
                    st.session_state.triage_queue = [q for q in st.session_state.triage_queue if q['id'] != p_id]
                    st.session_state.triage_queue.append(queue_entry)
                    
            st.success("Batch Processed! Check the Live Triage Board.")

# ------------------------------------------
# TAB 2: MANUAL INTAKE 
# ------------------------------------------
with tab2:
    st.header("Patient Intake Form")
    with st.form("patient_form"):
        p_id_manual = st.text_input("Patient ID", "P-MANUAL-01")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            arrival_mode = st.selectbox("Arrival Mode", ["walk_in", "ambulance", "wheelchair"])
            chronic = st.number_input("Chronic Diseases Count", 0, 10, 0)
        with c2:
            hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
            sbp = st.slider("Systolic BP", 70, 220, 120)
            prev_er = st.number_input("Previous ER Visits", 0, 20, 0)
        with c3:
            spo2 = st.slider("Oxygen Saturation (SpO2 %)", 70, 100, 98)
            temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
            pain = st.select_slider("Pain Level", options=list(range(1, 11)), value=5)

        submit_manual = st.form_submit_button("Analyze Patient")

        if submit_manual:
            manual_features = {
                'age': age, 'arrival_mode': arrival_mode, 'heart_rate': hr, 
                'systolic_blood_pressure': sbp, 'oxygen_saturation': spo2, 
                'body_temperature': temp, 'pain_level': pain, 
                'chronic_disease_count': chronic, 'previous_er_visits': prev_er,
                'raw_text': f"Patient arrived via {arrival_mode} complaining of {pain}/10 pain. Vitals: HR {hr}, SpO2 {spo2}."
            }
            
            with st.spinner("Processing AI Risk..."):
                # 1. Get ML Risk Level
                ai_result = process_triage_and_shap(manual_features, selected_features)
                t_level = ai_result.get('triage_level', 0)
                top_factors = ai_result.get('top_factors', [])
                shap_str = ", ".join([f"{f['feature']} ({f['direction']})" for f in top_factors])
                
                # 2. Get all 3 outputs from the LLM using the Delimiter Trick
                with st.spinner("Generating LLM Routing & Synthesis..."):
                    bio_dict = biomistral.get_explanation(t_level, shap_str, manual_features['raw_text'])
                
                # 3. History Tracking
                history_list = st.session_state.history.get(p_id_manual, [])
                trend = "stable"
                deltas = {"spo2": None, "temp": None, "hr": None}
                
                if len(history_list) > 0:
                    prev = history_list[-1]['features']
                    deltas['spo2'] = spo2 - prev.get('oxygen_saturation', 100)
                    deltas['temp'] = round(temp - prev.get('body_temperature', 37.0), 1)
                    deltas['hr'] = hr - prev.get('heart_rate', 80)
                    if spo2 < prev.get('oxygen_saturation', 100): trend = "worsening"

                st.session_state.history.setdefault(p_id_manual, []).append({"features": manual_features, "triage_level": t_level})
                
                # Map the Delimiter Dictionary out to the queue entry!
                queue_entry = {
                    "id": p_id_manual, "level": t_level, "trend": trend, 
                    "features": manual_features, "shap": shap_str, "deltas": deltas,
                    "bio_synthesis": bio_dict.get('short_synthesis', 'N/A'),
                    "bio_action": bio_dict.get('recommended_action', 'N/A'),
                    "bio_department": bio_dict.get('department_routing', 'N/A')
                }
                
                st.session_state.triage_queue = [q for q in st.session_state.triage_queue if q['id'] != p_id_manual]
                st.session_state.triage_queue.append(queue_entry)
                
            st.success(f"Patient {p_id_manual} prioritized at Level {t_level}! Check the Live Triage Board.")

# ------------------------------------------
# TAB 3: THE LIVE TRIAGE BOARD
# ------------------------------------------
with tab3:
    st.header("🚨 Live Triage Board")
    
    if not st.session_state.triage_queue:
        st.info("No patients processed yet. Upload PDFs or use the manual intake form.")
    else:
        sorted_queue = sorted(
            st.session_state.triage_queue, 
            key=lambda x: (x['level'], 1 if x['trend'] == 'worsening' else 0), 
            reverse=True
        )

        for patient in sorted_queue:
            if patient['level'] >= 2:
                card_color = "🔴 CRITICAL (Level 2/3)"
            elif patient['level'] == 1:
                card_color = "🟡 URGENT (Level 1)"
            else:
                card_color = "🟢 STABLE (Level 0)"

            with st.container():
                st.markdown(f"### {card_color} | Patient ID: {patient['id']}")
                
                if patient['trend'] == 'worsening':
                    st.error("⚠️ TREND ALERT: Patient condition is deteriorating compared to last visit.")
                
                m1, m2, m3, m4 = st.columns(4)
                
                d_spo2 = patient.get('deltas', {}).get('spo2')
                d_temp = patient.get('deltas', {}).get('temp')
                d_hr = patient.get('deltas', {}).get('hr')

                m1.metric(label="SpO2", value=f"{patient['features'].get('oxygen_saturation', 'N/A')}%", delta=f"{d_spo2}%" if d_spo2 is not None else None, delta_color="normal")
                m2.metric(label="Temp", value=f"{patient['features'].get('body_temperature', 'N/A')}°C", delta=f"{d_temp}°C" if d_temp is not None else None, delta_color="inverse")
                m3.metric(label="Heart Rate", value=f"{patient['features'].get('heart_rate', 'N/A')} BPM", delta=f"{d_hr} BPM" if d_hr is not None else None, delta_color="inverse")
                m4.metric(label="Pain", value=f"{patient['features'].get('pain_level', 'N/A')}/10")
                
                st.caption(f"**AI Reasoning (SHAP):** {patient['shap']}")
                
                # Unrestricted Output
                st.info(f"**🤖 Clinical Synthesis:** {patient['bio_synthesis']}")
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.error(f"**🚨 Recommended Action:**\n\n{patient['bio_action']}")
                with action_col2:
                    st.warning(f"**🏥 Route to Department:**\n\n{patient['bio_department']}")

                st.markdown("---")