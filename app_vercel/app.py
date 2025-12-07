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

# --- NASTAVENÍ CEST ---
BASE_DIR = "/tmp"  # Writable adresář na Vercelu (zde se ukládá stav session)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # Kořen projektu (zde jsou zdrojové kódy)
TABLES_SOURCE_DIR = os.path.join(PROJECT_DIR, 'tables') # Kde máš nahrané své zdrojové CSV (read-only)

# API Klient
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Modely (stejné jako v lokálním kódu)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.2
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"

# Kategorie - definice souborů
CATEGORY_FILES = {
    "DrABCDE": "DrABCDE.csv",
    "Medication": "Medication.csv",
    "Interventions": "Interventions.csv",
    "Physical Examination": "Physical Examination.csv", # Pozor na mezeru v názvu souboru, musí sedět s realitou
    "History": "History.csv",
    "SBAR": "SBAR.csv",
    "Other": "Other.csv" 
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vercel-secret'

# Globální paměť (pro warm-start Vercel instance)
data_tables = {}
item_mapping = {}
all_known_items = []
recent_updates = deque(maxlen=10)

# Výchozí stav vitálních funkcí
current_vitals = {
    "TF": "--", "TK": "--/--", "DF": "--", "SpO2": "--", "CRT": "--", "AVPU": "--"
}

# --- 1. KROK - SYNONYMA (Kopírováno z lokálního kódu) ---
ITEM_SYNONYMS = {
    # Tep -> Srdeční frekvence
    "tepová frekvence": "Srdeční frekvence", "typová frekvence": "Srdeční frekvence", 
    "tep": "Srdeční frekvence", "puls": "Srdeční frekvence", "sf": "Srdeční frekvence", 
    "tf": "Srdeční frekvence", "srdeční frekvence (sf)": "Srdeční frekvence",
    # Tlak -> Krevní tlak
    "tlak": "Krevní tlak", "tk": "Krevní tlak", "systolický tlak": "Krevní tlak",
    # Saturace -> SpO2
    "saturace": "SpO2", "sat": "SpO2", "spo2": "SpO2",
    # Dech -> Dechová frekvence
    "dech": "Dechová frekvence", "df": "Dechová frekvence",
    # Vědomí -> AVPU
    "vědomí": "AVPU", "gcs": "glasgow coma scale", "avpu": "AVPU", "glasgow coma scale": "gcs",
    # CRT
    "CRT": "Kapilární návrat", "crt": "Kapilární návrat"
}

# --- 2. KROK - MAPPING NA DISPLEJ (Kopírováno z lokálního kódu) ---
VITALS_MAPPING = {
    "srdeční frekvence": "TF", 
    "krevní tlak": "TK", 
    "dechová frekvence": "DF",
    "spo2": "SpO2", 
    "crt": "CRT", 
    "avpu": "AVPU"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_data_structures():
    """
    Načte data. Priorita:
    1. /tmp (pokud už jsme v rámci session něco uložili)
    2. ./tables (zdrojové čisté CSV)
    3. Vytvoří prázdné
    """
    global data_tables, item_mapping, all_known_items
    data_tables = {}
    item_mapping = {}
    all_known_items = []
    
    logging.info(f"Inicializace struktur. Zdroj: {TABLES_SOURCE_DIR}, Temp: {BASE_DIR}")

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for category, filename in CATEGORY_FILES.items():
        tmp_path = os.path.join(BASE_DIR, filename)       # /tmp/DrABCDE.csv
        source_path = os.path.join(TABLES_SOURCE_DIR, filename) # ./tables/DrABCDE.csv
        
        df = None
        
        # 1. Zkusíme načíst z /tmp (persistence v rámci Vercel instance)
        if os.path.exists(tmp_path):
            try:
                df = pd.read_csv(tmp_path, delimiter=';', index_col=0)
                logging.info(f"Načteno z TMP: {category}")
            except Exception as e:
                logging.warning(f"Chyba čtení TMP {filename}: {e}")

        # 2. Pokud není v TMP, zkusíme načíst originál
        if df is None and os.path.exists(source_path):
            try:
                df = pd.read_csv(source_path, delimiter=';', index_col=0)
                df.index.name = "Položka"
                # Zajistíme, že máme alespoň jeden sloupec pro data, pokud je CSV prázdné jen s indexem
                if df.empty and len(df.columns) == 0:
                     df['Aktuální stav'] = ""
                logging.info(f"Načteno ze SOURCE: {category} ({len(df)} položek)")
            except Exception as e:
                logging.error(f"Chyba čtení SOURCE {filename}: {e}")

        # 3. Fallback - pokud soubor neexistuje vůbec (např. Other.csv na začátku)
        if df is None:
            logging.info(f"Vytvářím prázdnou tabulku pro: {category}")
            df = pd.DataFrame(columns=['Aktuální stav'])
            df.index.name = "Položka"

        # Uložíme do globální proměnné
        data_tables[category] = df
        
        # Naplníme mapping pro AI (co patří do jaké kategorie)
        for item in df.index:
            item_str = str(item).strip()
            if item_str:
                item_mapping[item_str.lower()] = category
                all_known_items.append(item_str)

# Spustíme inicializaci při startu
initialize_data_structures()

# --- PROMPT SETUP (Dynamický) ---
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
    """Generuje HTML s historií. Upraveno pro skrytí kontextových sloupců."""
    tables_html = {}
    
    # Seznam klíčových slov v názvech sloupců, které chceme skrýt (metadata)
    # Tyto sloupce zůstávají v datech pro kontext LLM, ale nebudou v HTML.
    KEYWORDS_TO_HIDE = ["Popis", "Poznámka", "Referenční", "Norma", "Indikace", "Co a jak hodnotíme"]

    for category, df in data_tables.items():
        # Vytvoříme kopii pro zobrazení
        df_display = df.copy()
        
        # Identifikace sloupců k odstranění
        cols_to_drop = []
        for col in df_display.columns:
            # Pokud název sloupce obsahuje některé z klíčových slov pro metadata
            if any(keyword in col for keyword in KEYWORDS_TO_HIDE):
                cols_to_drop.append(col)
        
        # Odstranění sloupců z view (zůstane jen Index a dynamicky přidané časové sloupce)
        if cols_to_drop:
            df_display = df_display.drop(columns=cols_to_drop)

        # Zobrazíme prázdné buňky jako prázdné stringy
        tables_html[category] = df_display.fillna('').to_html(classes="table table-striped table-hover table-sm", border=0)
    return tables_html

def save_csvs_to_temp():
    """Uloží aktuální stav všech tabulek do /tmp."""
    try:
        if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)
        
        for category, df in data_tables.items():
            filename = CATEGORY_FILES.get(category)
            if filename:
                path = os.path.join(BASE_DIR, filename)
                df.to_csv(path, sep=';', encoding='utf-8')
    except Exception as e:
        logging.error(f"Save error: {e}")

@app.route('/')
def index():
    # Při každém načtení stránky se ujistíme, že máme data (pro případ restartu lambdy)
    if not data_tables:
        initialize_data_structures()
    return render_template('index.html', tables=generate_html_tables(), current_vitals=current_vitals, recent_updates=list(recent_updates))

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    # Ujistíme se, že máme inicializováno
    if not data_tables:
        initialize_data_structures()

    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio_file']
    
    try:
        # 1. Transkripce
        file.filename = "rec.wav"
        transcription = groq_client.audio.transcriptions.create(
            model=GROQ_TRANSCRIPTION_MODEL,
            file=(file.filename, file.read()),
            response_format="text",
            language="cs" 
        )
        text = transcription.strip()
        logging.info(f"Přepis: {text}")

        if not text or len(text) < 2:
            return jsonify({'transcription': '', 'extracted_data': {}, 'status': 'no_speech'})

        # 2. Extrakce dat pomocí LLM
        chat_completion = groq_client.chat.completions.create(
            messages=[{'role': 'system', 'content': get_system_prompt()}, {'role': 'user', 'content': text}],
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            response_format={"type": "json_object"}
        )
        extracted_data = json.loads(chat_completion.choices[0].message.content)
        
        # 3. Zpracování dat (Logic from Local Code)
        process_extracted_data(extracted_data)
        
        # 4. Uložení do /tmp
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

            # A) SYNONYMA (Sf -> Srdeční frekvence)
            if item_lower in ITEM_SYNONYMS:
                item_name = ITEM_SYNONYMS[item_lower]
                item_lower = item_name.lower()

            # B) VITALS DASHBOARD (Srdeční frekvence -> TF)
            vital_key = VITALS_MAPPING.get(item_lower)
            if vital_key: 
                current_vitals[vital_key] = str(value)

            # C) UPDATE HISTORIE
            recent_updates.appendleft(f"{timestamp} - {item_name} - {value}")
            
            # D) TABULKY
            category = item_mapping.get(item_lower)
            
            # Pokud kategorie nebyla nalezena přes mapping, zkusíme najít původní klíč
            if not category:
                 for known_item, cat in item_mapping.items():
                    if known_item == item_lower:
                        category = cat
                        # Získáme oficiální název (case sensitive) z tabulky
                        real_matches = data_tables[cat].index[data_tables[cat].index.str.lower() == known_item]
                        if not real_matches.empty:
                            item_name = real_matches[0]
                        break

            # Pokud stále nemáme kategorii -> OTHER
            if not category:
                category = "Other"
                if category not in data_tables:
                     # Inicializace Other tabulky, pokud neexistuje
                     data_tables["Other"] = pd.DataFrame(columns=['Aktuální stav'])
                     data_tables["Other"].index.name = "Položka"

                # Pokud položka v Other ještě není, přidáme ji
                if item_name not in data_tables["Other"].index:
                      # Vytvoříme řádek
                      new_cols = data_tables["Other"].columns
                      new_row = pd.DataFrame([[""] * len(new_cols)], columns=new_cols, index=[item_name])
                      data_tables["Other"] = pd.concat([data_tables["Other"], new_row])
                      
                      # Přidat do mappingu pro příště
                      item_mapping[item_lower] = "Other"
                      all_known_items.append(item_name)

            # E) ZÁPIS DO DATAFRAME
            df = data_tables[category]
            
            # Přidání sloupce s časem, pokud chybí
            if timestamp not in df.columns:
                df[timestamp] = ""

            # Zápis hodnoty
            if item_name in df.index:
                df.loc[item_name, timestamp] = str(value)
            else:
                 # Fallback pro case-insensitive hledání v indexu
                 matches = df.index[df.index.str.lower() == item_lower]
                 if not matches.empty:
                     df.loc[matches[0], timestamp] = str(value)

@app.route('/download_zip')
def download_zip():
    save_csvs_to_temp()
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for category, filename in CATEGORY_FILES.items():
            file_path = os.path.join(BASE_DIR, filename)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=filename)
                
    memory_file.seek(0)
    return send_file(
        memory_file, 
        mimetype='application/zip', 
        as_attachment=True, 
        download_name=f'data_{datetime.now().strftime("%H%M")}.zip'
    )

@app.route('/reset', methods=['POST'])
def reset_session():
    # Smažeme soubory v /tmp
    for filename in CATEGORY_FILES.values():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            
    # Reset paměti a znovu načtení ze SOURCE
    global current_vitals, recent_updates
    current_vitals = {k: "--" for k in current_vitals}
    recent_updates.clear()
    
    initialize_data_structures() 
    
    return jsonify({'status': 'ok'})

@app.route('/sw.js')
def service_worker():
    response = send_from_directory('static', 'sw.js')
    # Zakážeme cacheování samotného souboru sw.js, aby se změny projevily hned
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

if __name__ == '__main__':
    app.run(debug=True, port=5050)