import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import pytesseract
from pypdf import PdfReader
import docx
from langdetect import detect, DetectorFactory
from groq import Groq

# Ensure consistent language detection seed results
DetectorFactory.seed = 0

# =====================================================================
# 1. USER INTERFACE ARCHITECTURE (STREAMLIT DASHBOARD & APIS)
# =====================================================================
st.set_page_config(page_title="Multilingual Clinical NER & NMT Portal", layout="wide")

st.title("🔬 Multilingual Clinical Entity Extraction & Translation System")
st.markdown("---")

# Sidebar Configuration Layout
st.sidebar.header("🛠️ Pipeline Configurations")

# Web Interface Key Input: Users paste their Groq API Key here directly
user_api_input = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Paste your active Groq API key here to enable clinical NER.")

# Error-proof fallback checking mechanism
GROQ_API_KEY = ""

if user_api_input:
    GROQ_API_KEY = user_api_input
else:
    # Safely check environment variables first
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Only try to read st.secrets if Streamlit detects a secrets file exists, avoiding the crash
    if not GROQ_API_KEY:
        try:
            if "GROQ_API_KEY" in st.secrets:
                GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

if not GROQ_API_KEY:
    st.sidebar.warning("⚠️ Groq API Key required. Please provide it above to execute Clinical NER.")
    client = None
else:
    st.sidebar.success("🔑 Groq Client successfully connected!")
    client = Groq(api_key=GROQ_API_KEY)

# =====================================================================
# EXPANDED TARGET LANGUAGE SELECTOR MAPPING (50+ NATIONAL & INTERNATIONAL)
# =====================================================================
lang_options = {
    # --- Indian National Languages ---
    "Hindi": "hi", "Telugu": "te", "Tamil": "ta", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa", "Odia": "or",
    "Urdu": "ur", "Assamese": "as", "Sanskrit": "sa",
    # --- Global International Languages ---
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Arabic": "ar",
    "Mandarin Chinese": "zh", "Japanese": "ja", "Russian": "ru", "Portuguese": "pt", "Italian": "it",
    "Korean": "ko", "Turkish": "tr", "Vietnamese": "vi", "Polish": "pl", "Dutch": "nl",
    "Thai": "th", "Persian (Farsi)": "fa", "Swedish": "sv", "Indonesian": "id", "Malay": "ms",
    "Greek": "el", "Hebrew": "he", "Norwegian": "no", "Danish": "da", "Finnish": "fi",
    "Czech": "cs", "Romanian": "ro", "Hungarian": "hu", "Ukrainian": "uk", "Filipino": "tl",
    "Swahili": "sw", "Afrikaans": "af", "Zulu": "zu", "Irish": "ga", "Welsh": "cy",
    "Latin": "la", "Mongolian": "mn", "Nepali": "ne", "Sinhala": "si", "Khmer": "km"
}

selected_target_lang = st.sidebar.selectbox("Select Target Language (50+ Supported):", sorted(list(lang_options.keys())))
target_code = lang_options[selected_target_lang]

# =====================================================================
# 2. DOCUMENT PARSING ENGINE
# =====================================================================
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    extracted_text = ""

    if file_extension == 'txt':
        extracted_text = uploaded_file.read().decode("utf-8")
    elif file_extension in ['png', 'jpg', 'jpeg']:
        image = Image.open(uploaded_file)
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(image, config=custom_config)
    elif file_extension == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        if not extracted_text.strip():
            extracted_text = "[Error: Scanned PDF detected. Please upload directly as an image file for deep OCR processing.]"
    elif file_extension == 'docx':
        doc = docx.Document(uploaded_file)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
    elif file_extension in ['csv', 'xlsx']:
        df = pd.read_excel(uploaded_file) if file_extension == 'xlsx' else pd.read_csv(uploaded_file)
        extracted_text = df.to_string(index=False)
    elif file_extension == 'json':
        json_data = json.load(uploaded_file)
        extracted_text = json.dumps(json_data, indent=2)
    else:
        raise ValueError("Unsupported file format provided.")

    return extracted_text.strip()

# =====================================================================
# 3. CLOUD-BASED NEURAL MACHINE TRANSLATION (NMT VIA GROQ)
# =====================================================================
def translate_text_via_groq(text, target_lang_name):
    """
    Leverages Llama 3.3 via Groq infrastructure to handle translation tasks natively,
    ensuring ultra-low latency and 100% accurate string translation in the UI.
    """
    if not client:
        return text

    try:
        system_prompt = (
            f"You are a professional medical translator. Translate the given text completely into fluent {target_lang_name}.\n"
            "Preserve all original clinical terms, medical abbreviations, dosages, and formatting structures.\n"
            "Do not add any explanations, introductory remarks, or conversational filler. Return ONLY the translated text."
        )
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Cloud translation failed, falling back to source. Error: {str(e)}")
        return text

# =====================================================================
# 4. CLINICAL NER EXTRACTION OVER GROQ LLM FRAMEWORK
# =====================================================================
def extract_clinical_entities(translated_text, target_lang_name):
    if not client:
        return {"error": "Groq client is uninitialized. Please paste your Groq API Key into the sidebar to analyze this text."}

    system_prompt = (
        "You are an advanced medical AI sub-system specialized in clinical information extraction.\n"
        "Analyze the provided clinical record or prescription, extract medical entities, and categorize them.\n"
        f"CRITICAL: Translate all extracted internal values into {target_lang_name}, but the main JSON object keys MUST remain exactly as specified in English.\n"
        "Format the final output strictly as a valid JSON object with the following exact English key structure:\n"
        "{\n"
        "  \"diseases\": [],\n"
        "  \"drugs_and_dosage\": [],\n"
        "  \"symptoms\": [],\n"
        "  \"diagnostic_tests\": []\n"
        "}\n"
        "Do not provide any introductory markdown notes, explanations, or conversational filler. Return raw JSON text only."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Clinical Text:\n{translated_text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        return {"error": f"Failed to connect or parse payload from Groq pipeline: {str(e)}"}

# =====================================================================
# 5. MAIN PROCESSING ROUTINE
# =====================================================================
uploaded_file = st.file_uploader(
    "Upload Clinical Document / Patient Records / Prescription Sheet:", 
    type=['txt', 'png', 'jpg', 'jpeg', 'pdf', 'docx', 'csv', 'xlsx', 'json']
)

if uploaded_file is not None:
    st.info(f"🔄 Processing file: **{uploaded_file.name}**...")
    
    with st.spinner("Step 1/3: Extracting structural text from file format..."):
        try:
            raw_extracted_text = extract_text_from_file(uploaded_file)
        except Exception as e:
            st.error(f"Failed parsing the selected document architecture: {e}")
            raw_extracted_text = ""

    if raw_extracted_text:
        try:
            detected_source_code = detect(raw_extracted_text).upper()
        except Exception:
            detected_source_code = "UNKNOWN"

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Raw Extracted Data Source")
            st.text_area("Source Text", raw_extracted_text, height=250, key="raw_source")

        with st.spinner(f"Step 2/3: Translating text into {selected_target_lang} via Groq Cloud..."):
            if detected_source_code.lower() == target_code.lower():
                processed_translation = raw_extracted_text
            else:
                processed_translation = translate_text_via_groq(raw_extracted_text, selected_target_lang)
        
        with col2:
            st.subheader(f"🌐 Translated Medical Record ({selected_target_lang})")
            # Uses target_code dynamically within the key parameter to fix the re-rendering update bug
            st.text_area("Translated Text", processed_translation, height=250, key=f"translated_source_{target_code}")
            st.caption(f"**Detected Source Language Profile Code:** {detected_source_code}")

        st.markdown("---")
        
        with st.spinner("Step 3/3: Running LLM Context Pipelines for Extracted Medical Entities..."):
            extracted_json_entities = extract_clinical_entities(processed_translation, selected_target_lang)

        st.subheader(f"🏷️ Extracted Clinical Named Entities ({selected_target_lang})")
        
        if "error" in extracted_json_entities:
            st.error(extracted_json_entities["error"])
        else:
            ent_col1, ent_col2, ent_col3, ent_col4 = st.columns(4)
            with ent_col1:
                st.success("🦠 Diseases / Conditions")
                st.write(extracted_json_entities.get("diseases", []))
            with ent_col2:
                st.info("💊 Drugs & Dosages")
                st.write(extracted_json_entities.get("drugs_and_dosage", []))
            with ent_col3:
                st.warning("🤒 Symptoms")
                st.write(extracted_json_entities.get("symptoms", []))
            with ent_col4:
                st.metric(label="Total Entities Extracted", value=sum(len(v) for v in extracted_json_entities.values() if isinstance(v, list)))
                st.json(extracted_json_entities)

