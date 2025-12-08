import os
import json
import logging
import pandas as pd
from groq import Groq
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from datetime import datetime
from collections import deque
import zipfile
import io
import google.generativeai as genai

# --- NASTAVENÍ CEST ---
BASE_DIR = "/tmp"  # Writable adresář na Vercelu
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_SOURCE_DIR = os.path.join(PROJECT_DIR, 'tables')

# API Klient - GROQ (Whisper)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# API Klient - GOOGLE GEMINI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- KONFIGURACE MODELŮ ---
# Defaultní provider při startu
CURRENT_LLM_PROVIDER = "GEMINI" 

# Definice modelů
MODELS_CONFIG = {
    "GROQ_LLAMA": "llama-3.1-8b-instant",
    "GROQ_WHISPER": "whisper-large-v3-turbo",
    "GEMINI": "gemini-1.5-flash", #gemini-1.5-flash gemini-2.5-flash gemini-2.5-flash-lite
    "TEMPERATURE": 0.1
}

# Kategorie - definice souborů
CATEGORY_FILES = {
    "DrABCDE": "DrABCDE.csv",
    "Medication": "Medication.csv",
    "Interventions": "Interventions.csv",
    "Physical Examination": "Physical Examination.csv",
    "History": "History.csv",
    "SBAR": "SBAR.csv",
    "Other": "Other.csv" 
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vercel-secret'

# Globální paměť
data_tables = {}
item_mapping = {}
all_known_items = []
recent_updates = deque(maxlen=10)
# NOVÉ: Log pro raw přepisy
raw_transcripts_log = [] 

# Výchozí stav vitálních funkcí
current_vitals = {
    "TF": "--", "TK": "--/--", "DF": "--", "SpO2": "--", "CRT": "--", "AVPU": "--"
}

# --- 1. KROK - SYNONYMA ---
ITEM_SYNONYMS = {
    "tepová frekvence": "Srdeční frekvence", "typová frekvence": "Srdeční frekvence", 
    "tep": "Srdeční frekvence", "puls": "Srdeční frekvence", "sf": "Srdeční frekvence", 
    "tf": "Srdeční frekvence", "srdeční frekvence (sf)": "Srdeční frekvence",
    "tlak": "Krevní tlak", "tk": "Krevní tlak", "systolický tlak": "Krevní tlak",
    "saturace": "SpO2", "sat": "SpO2", "spo2": "SpO2",
    "dech": "Dechová frekvence", "df": "Dechová frekvence",
    "vědomí": "AVPU", "gcs": "glasgow coma scale", "avpu": "AVPU", "glasgow coma scale": "gcs",
    "CRT": "Kapilární návrat", "crt": "Kapilární návrat"
}

# --- 2. KROK - MAPPING NA DISPLEJ ---
VITALS_MAPPING = {
    "srdeční frekvence": "TF", 
    "krevní tlak": "TK", 
    "dechová frekvence": "DF",
    "spo2": "SpO2", 
    "kapilární návrat": "CRT", 
    "avpu": "AVPU"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_data_structures():
    global data_tables, item_mapping, all_known_items
    data_tables = {}
    item_mapping = {}
    all_known_items = []
    
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for category, filename in CATEGORY_FILES.items():
        tmp_path = os.path.join(BASE_DIR, filename)
        source_path = os.path.join(TABLES_SOURCE_DIR, filename)
        df = None
        
        # 1. Zkusíme načíst z /tmp
        if os.path.exists(tmp_path):
            try:
                df = pd.read_csv(tmp_path, delimiter=';', index_col=0)
            except Exception: pass

        # 2. Zkusíme načíst originál
        if df is None and os.path.exists(source_path):
            try:
                df = pd.read_csv(source_path, delimiter=';', index_col=0)
                df.index.name = "Položka"
                if df.empty and len(df.columns) == 0: df['Aktuální stav'] = ""
            except Exception: pass

        # 3. Fallback
        if df is None:
            df = pd.DataFrame(columns=['Aktuální stav'])
            df.index.name = "Položka"

        data_tables[category] = df
        
        for item in df.index:
            item_str = str(item).strip()
            if item_str:
                item_mapping[item_str.lower()] = category
                all_known_items.append(item_str)

initialize_data_structures()

# --- PROMPT SETUP ---
def get_system_prompt():
    items_list_str = ', '.join(f'"{item}"' for item in all_known_items)
    return f"""
Jsi expert na extrakci lékařských dat z mluveného slova v reálném čase.
Tvým úkolem je naslouchat hlášením lékaře/záchranáře a strukturovat je do JSON formátu.
Údaje si NEVYMYŠLEJ, pokud nejsou explicitně uvedeny v textu, ponech je prázdné.
Pracuj pouze s textem v českém jazyce.

Pravidla:
1. Výstup musí být VŽDY a POUZE platný JSON objekt.
2. JSON obsahuje klíč "polozky", což je slovník.
3. Klíče ve slovníku "polozky" musí odpovídat názvům z následujícího seznamu (nebo jejich synonymům, které normalizuješ):
   [{items_list_str}]
4. Normalizuj synonyma na oficiální názvy (např. "tep" -> "Srdeční frekvence", "saturace" -> "SpO2").
5. Pokud narazíš na položku, která v seznamu není, ale je lékařsky relevantní, zahrň ji také pod jejím obvyklým názvem.
6. Hodnoty extrahuj jako čísla nebo formátovaný string (např. "120/80").

Příklady:
Uživatel: "Pacient má saturaci 92 a tlak 130 na 80."
Výstup: {{"polozky": {{"SpO2": "92", "Krevní tlak": "130/80"}}}}
"""

def generate_html_tables():
    tables_html = {}
    KEYWORDS_TO_HIDE = ["Popis", "Poznámka", "Referenční", "Norma", "Indikace", "Co a jak hodnotíme"]

    for category, df in data_tables.items():
        df_display = df.copy()
        cols_to_drop = []
        for col in df_display.columns:
            if any(keyword in col for keyword in KEYWORDS_TO_HIDE):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_display = df_display.drop(columns=cols_to_drop)

        tables_html[category] = df_display.fillna('').to_html(classes="table table-striped table-hover table-sm", border=0)
    return tables_html

def save_csvs_to_temp():
    try:
        if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)
        for category, df in data_tables.items():
            filename = CATEGORY_FILES.get(category)
            if filename:
                path = os.path.join(BASE_DIR, filename)
                df.to_csv(path, sep=';', encoding='utf-8')
    except Exception as e:
        logging.error(f"Save error: {e}")

# --- ROUTES ---

@app.route('/')
def index():
    if not data_tables: initialize_data_structures()
    return render_template('index.html', 
                           tables=generate_html_tables(), 
                           current_vitals=current_vitals, 
                           recent_updates=list(recent_updates),
                           current_provider=CURRENT_LLM_PROVIDER)

@app.route('/api/set_provider', methods=['POST'])
def set_provider():
    """Endpoint pro přepínání LLM modelu z UI"""
    global CURRENT_LLM_PROVIDER
    data = request.json
    new_provider = data.get('provider')
    if new_provider in ["GEMINI", "GROQ"]:
        CURRENT_LLM_PROVIDER = new_provider
        logging.info(f"Provider switched to: {CURRENT_LLM_PROVIDER}")
        return jsonify({'status': 'success', 'provider': CURRENT_LLM_PROVIDER})
    return jsonify({'status': 'error', 'message': 'Invalid provider'}), 400

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    if not data_tables: initialize_data_structures()
    if 'audio_file' not in request.files: return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio_file']
    
    try:
        # 1. Transkripce (Whisper)
        file.filename = "rec.wav"
        transcription = groq_client.audio.transcriptions.create(
            model=MODELS_CONFIG["GROQ_WHISPER"],
            file=(file.filename, file.read()),
            response_format="text",
            language="cs" 
        )
        text = transcription.strip()
        logging.info(f"Přepis: {text}")

        if not text or len(text) < 2:
            return jsonify({'transcription': '', 'extracted_data': {}, 'status': 'no_speech'})

        # 2. Extrakce dat
        extracted_data = {}
        
        if CURRENT_LLM_PROVIDER == "GEMINI":
            logging.info(f"Processing via GEMINI ({MODELS_CONFIG['GEMINI']})")
            model = genai.GenerativeModel(
                model_name=MODELS_CONFIG["GEMINI"],
                system_instruction=get_system_prompt()
            )
            response = model.generate_content(text, generation_config={"response_mime_type": "application/json"})
            extracted_data = json.loads(response.text)
            
        else: # GROQ
            logging.info(f"Processing via GROQ ({MODELS_CONFIG['GROQ_LLAMA']})")
            chat_completion = groq_client.chat.completions.create(
                messages=[{'role': 'system', 'content': get_system_prompt()}, {'role': 'user', 'content': text}],
                model=MODELS_CONFIG["GROQ_LLAMA"],
                temperature=MODELS_CONFIG["TEMPERATURE"],
                response_format={"type": "json_object"}
            )
            extracted_data = json.loads(chat_completion.choices[0].message.content)
        
        # --- ZMĚNA: LOGOVÁNÍ AŽ PO EXTRAKCI JSONU ---
        timestamp_str = datetime.now().strftime('%H:%M:%S')
        # Formátování záznamu pro debug (obsahuje text i JSON)
        log_entry = (
            f"[{timestamp_str}]\n"
            f"TRANSCRIPTION: {text}\n"
            f"EXTRACTED JSON: {json.dumps(extracted_data, ensure_ascii=False)}\n"
            f"{'-'*40}"
        )
        raw_transcripts_log.append(log_entry)
        
        # 3. Update dat
        process_extracted_data(extracted_data)
        save_csvs_to_temp()

        return jsonify({
            'transcription': text,
            'extracted_data': extracted_data,
            'tables': generate_html_tables(),
            'current_vitals': current_vitals,
            'recent_updates': list(recent_updates),
            'status': 'success'
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

def process_extracted_data(data):
    global current_vitals
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if 'polozky' in data:
        for item_name, value in data['polozky'].items():
            item_lower = item_name.lower()
            if item_lower in ITEM_SYNONYMS:
                item_name = ITEM_SYNONYMS[item_lower]
                item_lower = item_name.lower()

            vital_key = VITALS_MAPPING.get(item_lower)
            if vital_key: current_vitals[vital_key] = str(value)

            recent_updates.appendleft(f"{timestamp} - {item_name} - {value}")
            
            category = item_mapping.get(item_lower)
            if not category:
                 for known_item, cat in item_mapping.items():
                    if known_item == item_lower:
                        category = cat
                        real_matches = data_tables[cat].index[data_tables[cat].index.str.lower() == known_item]
                        if not real_matches.empty: item_name = real_matches[0]
                        break

            if not category:
                category = "Other"
                if category not in data_tables:
                     data_tables["Other"] = pd.DataFrame(columns=['Aktuální stav'])
                     data_tables["Other"].index.name = "Položka"

                if item_name not in data_tables["Other"].index:
                      new_cols = data_tables["Other"].columns
                      new_row = pd.DataFrame([[""] * len(new_cols)], columns=new_cols, index=[item_name])
                      data_tables["Other"] = pd.concat([data_tables["Other"], new_row])
                      item_mapping[item_lower] = "Other"
                      all_known_items.append(item_name)

            df = data_tables[category]
            if timestamp not in df.columns: df[timestamp] = ""

            if item_name in df.index: df.loc[item_name, timestamp] = str(value)
            else:
                 matches = df.index[df.index.str.lower() == item_lower]
                 if not matches.empty: df.loc[matches[0], timestamp] = str(value)

@app.route('/download_zip')
def download_zip():
    save_csvs_to_temp()
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Přidat CSV soubory
        for category, filename in CATEGORY_FILES.items():
            file_path = os.path.join(BASE_DIR, filename)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=filename)
        
        # 2. Přidat Debug Info
        debug_info = f"""DEBUG LOG
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current Provider: {CURRENT_LLM_PROVIDER}
Models Config:
{json.dumps(MODELS_CONFIG, indent=2)}
"""
        zf.writestr("debug_info.txt", debug_info)

        # 3. Přidat Raw Transcripts + JSON
        raw_text_content = "RAW TRANSCRIPTION & EXTRACTION LOG\n==================================\n\n"
        raw_text_content += "\n".join(raw_transcripts_log)
        zf.writestr("raw_transcripts.txt", raw_text_content)
                
    memory_file.seek(0)
    return send_file(
        memory_file, 
        mimetype='application/zip', 
        as_attachment=True, 
        download_name=f'data_{datetime.now().strftime("%H%M")}.zip'
    )

@app.route('/reset', methods=['POST'])
def reset_session():
    global current_vitals, recent_updates, raw_transcripts_log
    
    for filename in CATEGORY_FILES.values():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path): os.remove(path)
            
    current_vitals = {k: "--" for k in current_vitals}
    recent_updates.clear()
    raw_transcripts_log.clear() # Vymazání logu přepisů
    
    initialize_data_structures() 
    return jsonify({'status': 'ok'})

@app.route('/sw.js')
def service_worker():
    response = send_from_directory('static', 'sw.js')
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

if __name__ == '__main__':
    app.run(debug=True, port=5050)