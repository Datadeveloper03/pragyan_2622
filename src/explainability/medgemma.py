import requests

class BioMistralExplainer:
    def __init__(self, model_name="adrienbrault/biomistral-7b:Q4_K_M"):
        self.url = "http://127.0.0.1:11434/api/generate"
        self.model = model_name

    def get_explanation(self, triage_level, shap_info, symptoms):
        safe_symptoms = symptoms[:500].replace('\n', ' ') + "..." if len(symptoms) > 500 else symptoms.replace('\n', ' ')

        # THE DELIMITER PROMPT: We force it to use '|||' to separate the answers.
        # This is the most reliable method for 7B models.
        prompt = f"""[INST] You are an AI Chief Medical Officer. Analyze the patient data and return exactly ONE line of text.
You MUST separate your 3 answers using the '|||' symbol.

Format:
Clinical Synthesis (1 sentence) ||| Recommended Action (3-5 words) ||| Department Routing (1-3 words)

Example:
Patient is a 55-year-old male presenting with severe chest pain and tachycardia. ||| Stat EKG and Troponin ||| Cardiac ICU

Patient Data:
- Risk Level: {triage_level}
- AI Drivers: {shap_info}
- Notes: {safe_symptoms}
[/INST]
"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.1, # Keep it low so it doesn't get creative with the formatting
                "num_predict": 100, 
                "num_ctx": 2048
            }
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            
            if response.status_code == 200:
                raw_output = response.json().get('response', "").strip()
                
                # Print to your terminal so you can see exactly what it generated!
                print(f"\n--- RAW LLM OUTPUT ---\n{raw_output}\n----------------------\n")
                
                # Flatten the string just in case the model added accidental line breaks
                raw_output = raw_output.replace('\n', ' ')
                
                # THE FOOLPROOF PARSER: Split the string by the '|||' symbol
                parts = raw_output.split('|||')
                
                if len(parts) >= 3:
                    return {
                        "short_synthesis": parts[0].strip(),
                        "recommended_action": parts[1].strip(),
                        "department_routing": parts[2].strip()
                    }
                elif len(parts) == 2:
                    return {
                        "short_synthesis": parts[0].strip(),
                        "recommended_action": parts[1].strip(),
                        "department_routing": "General Triage"
                    }
                else:
                    # If it completely failed the format, just dump whatever it wrote into the synthesis
                    return {
                        "short_synthesis": raw_output[:150],
                        "recommended_action": "Manual Review",
                        "department_routing": "General Triage"
                    }
                    
            return {"short_synthesis": f"API Error {response.status_code}", "recommended_action": "Error", "department_routing": "Error"}
            
        except requests.exceptions.ReadTimeout:
            return {"short_synthesis": "Model Timeout.", "recommended_action": "Retry", "department_routing": "Timeout"}
        except requests.exceptions.ConnectionError:
            return {"short_synthesis": "Connection Failed.", "recommended_action": "Start Ollama", "department_routing": "Offline"}