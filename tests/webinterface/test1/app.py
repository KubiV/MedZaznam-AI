import ollama
import json
import pandas as pd
import logging
import os
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

# --- LOGGING SETUP ---
# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create logs directory with proper error handling
try:
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logs directory created/verified at: {LOGS_DIR}")
except Exception as e:
    print(f"Error creating logs directory: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'inventory.log')),
        logging.StreamHandler()
    ]
)

logging.info(f"Logging initialized. Log directory: {LOGS_DIR}")

# --- INICIALIZACE APLIKACE A DAT ---

app = Flask(__name__)

# Seznam výchozích položek
PREDEFINED_ITEMS = [
    'Mléko', 'Chléb', 'Máslo', 'Sýr', 'Vejce', 'Jogurt', 'Ovoce', 'Zelenina',
    'Brambory', 'Těstoviny', 'Rýže', 'Maso', 'Uzeniny', 'Káva', 'Čaj', 'Cukr',
    'Mouka', 'Olej', 'Toaletní papír', 'Mýdlo', 'Prací prostředek', 'Rohlík', 'Houska'
]
# Zajistíme unikátnost a správné pojmenování
PREDEFINED_ITEMS = [item.strip().capitalize() for item in PREDEFINED_ITEMS]

# Globální DataFrame, který bude držet stav naší tabulky
df_state = pd.DataFrame(
    {'Počáteční stav': 0},
    index=PREDEFINED_ITEMS
)
df_state.index.name = "Položka"


# --- LOGIKA PRO KOMUNIKACI S OLLAMA ---

# OPRAVA: Definice systémového promptu je nyní zde, přímo nad funkcí, která ji používá.
system_prompt_pro_flask = """
Jsi expert na extrakci dat pro systém sledování zásob. Z textu extrahuj potraviny, jejich počet a typ operace.

Pravidla:
1.  Výstup musí být VŽDY a POUZE platný JSON objekt.
2.  JSON obsahuje klíč 'operace', který může mít dvě hodnoty:
    - 'prirustek': Pokud text popisuje přidání nebo odebrání položek (např. "koupil jsem", "přidal jsem", "vrátil jsem"). Zde použij záporná čísla pro odebrání.
    - 'nastaveni': Pokud text popisuje finální, absolutní stav (např. "mám celkem", "zůstalo mi", "aktuální stav je").
3.  Druhý klíč je 'polozky', což je slovník, kde klíče jsou názvy potravin v 1. pádu jednotného čísla a hodnoty jsou čísla.

Příklady:
Uživatel: "Koupil jsem 2 mléka a 3 rohlíky. Musel jsem vrátit jednu housku."
Tvůj výstup:
{
  "operace": "prirustek",
  "polozky": {
    "mléko": 2,
    "rohlík": 3,
    "houska": -1
  }
}

Uživatel: "Po inventuře mám v košíku už jen 1 rohlík a 5 vajec."
Tvůj výstup:
{
  "operace": "nastaveni",
  "polozky": {
    "rohlík": 1,
    "vejce": 5
  }
}
"""

def get_data_from_ollama(text: str) -> dict | None:
    """Odešle text do Ollamy a vrátí strukturovaná data."""
    logging.info(f"Sending text to Ollama: {text}")
    try:
        response = ollama.chat(
            model='gemma3:1b', # nebo jiný váš model gemma3:4b gemma3:1b deepseek-r1:1.5b
            messages=[
                # Zde se proměnná používá
                {'role': 'system', 'content': system_prompt_pro_flask},
                {'role': 'user', 'content': text}
            ],
            options={'response_format': {'type': 'json_object'}}
        )

        json_text = response['message']['content']
        start = json_text.find('{')
        end = json_text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(json_text[start:end])
        return None
    except Exception as e:
        logging.error(f"Error in Ollama communication: {e}")
        return None

# --- FLASK ROUTY (WEBOVÉ ROZHRANÍ) ---

@app.route('/')
def index():
    """Zobrazí hlavní stránku s tabulkou."""
    logging.info("Index page accessed")
    table_html = df_state.to_html(classes="table table-striped table-hover", border=0)
    return render_template('index.html', table=table_html)

@app.route('/process', methods=['POST'])
def process_text():
    """Zpracuje odeslaný text, aktualizuje tabulku a přesměruje zpět na hlavní stránku."""
    global df_state, PREDEFINED_ITEMS
    user_text = request.form['text_input']
    logging.info(f"Processing new input: {user_text}")

    if not user_text:
        logging.warning("Empty input received")
        return redirect(url_for('index'))

    extracted_data = get_data_from_ollama(user_text)

    if extracted_data and 'operace' in extracted_data and 'polozky' in extracted_data:
        logging.info(f"Successfully extracted data: {extracted_data}")
        timestamp = datetime.now().strftime('%H:%M:%S')
        last_column = df_state.columns[-1]
        df_state[timestamp] = df_state[last_column].copy()

        operation = extracted_data['operace']
        items = extracted_data['polozky']

        for item_name, value in items.items():
            item_name_clean = item_name.strip().capitalize()

            if item_name_clean not in df_state.index:
                # Add to PREDEFINED_ITEMS if not present
                if item_name_clean not in PREDEFINED_ITEMS:
                    PREDEFINED_ITEMS.append(item_name_clean)

                # Create new row initialized with zeros for all columns
                new_row = pd.Series(0, index=df_state.columns)
                df_state.loc[item_name_clean] = new_row

            if operation == 'prirustek':
                df_state.loc[item_name_clean, timestamp] = df_state.loc[item_name_clean, last_column] + value
            elif operation == 'nastaveni':
                df_state.loc[item_name_clean, timestamp] = value

    else:
        logging.warning("Failed to extract data from input")

    return redirect(url_for('index'))

if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True)