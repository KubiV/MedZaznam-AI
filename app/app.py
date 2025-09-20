import ollama
from groq import Groq
import json
import pandas as pd
import logging
import os
import threading
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
from datetime import datetime
from collections import deque
import atexit
import tempfile
import wave

# --- NASTAVENÍ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
# Ujistěte se, že cesta k Vosk modelu je správná
#MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-cs-0.4-rhasspy") 
MODEL_PATH = "/Users/jakubvavra/Documents/GitHub/Automonitoring-with-AI/tests/vosk/vosk-model-small-cs-0.4-rhasspy"
CSV_DATA_PATH = os.path.join(BASE_DIR, 'Hodnoty MED.csv')

# Vytvoření adresářů
os.makedirs(LOGS_DIR, exist_ok=True)

# Nastavení logování
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'medical_app.log')),
        logging.StreamHandler()
    ]
)

# --- AUDIO PARAMETRY ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 150
MIN_SPEECH_DURATION = 0.5  # V sekundách
SILENCE_DURATION = 1.5     # V sekundách
MAX_RECORDING_TIME = 20    # V sekundách

# --- INICIALIZACE APLIKACE ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- GLOBÁLNÍ PROMĚNNÉ A ZÁMKY ---
AI_PROVIDER = 'local'  # Výchozí hodnota: 'local' nebo 'groq'
speech_processor = None
is_recording = False
recording_lock = threading.Lock()

# --- NAČTENÍ A PŘÍPRAVA DAT ---
try:
    df_med_items = pd.read_csv(CSV_DATA_PATH, delimiter=';')
    # -----------------------------------------------------------------
    # OPRAVA: Přejmenování sloupce 'DrABCDE' na 'Položka'
    # -----------------------------------------------------------------
    df_med_items.rename(columns={'DrABCDE': 'Položka'}, inplace=True)       
    
    numeric_items = []
    text_items = []
    
    for _, row in df_med_items.iterrows():
        # Zkontrolujeme, zda hodnota v 'Referenční hodnoty' není prázdná
        if pd.notna(row['Referenční hodnoty']):
            ref_val = str(row['Referenční hodnoty'])
            # Jednoduchá logika: pokud referenční hodnota obsahuje číslici, je to číselná položka
            if any(char.isdigit() for char in ref_val):
                numeric_items.append(row['Položka'])
            else:
                text_items.append(row['Položka'])
        else:
            # Pokud je referenční hodnota prázdná, předpokládáme textovou položku
            text_items.append(row['Položka'])


    df_numeric_state = pd.DataFrame({'Počáteční stav': np.nan}, index=numeric_items)
    df_text_state = pd.DataFrame({'Počáteční stav': ''}, index=text_items)
    df_numeric_state.index.name = "Položka"
    df_text_state.index.name = "Položka"

except FileNotFoundError:
    logging.error(f"Soubor s daty nebyl nalezen: {CSV_DATA_PATH}. Používám záložní data.")
    numeric_items = ['SpO2', 'Srdeční frekvence', 'Krevní tlak', 'Dechová frekvence', 'Teplota', 'Bolest NRS']
    text_items = ['Dušnost', 'Kyslíková terapie', 'Léky', 'Anamnéza', 'Fyzikální vyšetření']
    df_numeric_state = pd.DataFrame({'Počáteční stav': np.nan}, index=numeric_items)
    df_text_state = pd.DataFrame({'Počáteční stav': ''}, index=text_items)


# --- LLM PROMPTY ---
numeric_items_str = ', '.join(f'"{item}"' for item in numeric_items)
text_items_str = ', '.join(f'"{item}"' for item in text_items)

system_prompt = f"""
Jsi expert na extrakci lékařských dat z mluveného slova. Tvým úkolem je z textu extrahovat medicínské parametry a nálezy.

Pravidla:
1.  Výstup musí být VŽDY a POUZE platný JSON objekt.
2.  JSON obsahuje klíč "polozky", což je slovník.
3.  Klíče ve slovníku "polozky" musí být POUZE názvy z jednoho z těchto dvou seznamů:
    - Číselné položky: [{numeric_items_str}]
    - Textové položky: [{text_items_str}]
4.  Normalizuj synonymum na oficiální název (např. "tep" -> "Srdeční frekvence", "saturace" -> "SpO2", "tlak" -> "Krevní tlak"). Dbej na velikost písmen.
5.  Hodnoty:
    - Pro číselné položky extrahuj POUZE číslo (např. z "saturace 98 procent" extrahuj `98`). Pokud je tlak, vrať "systola/diastola" (např. "120/80").
    - Pro textové položky extrahuj stručný a relevantní popis (např. z "pacient si stěžuje na dušnost při námaze" extrahuj `"při námaze"`).
6.  Pokud text neobsahuje žádné relevantní informace, vrať prázdný slovník `{{"polozky": {{}}}}`.

Příklady:
Uživatel: "Pacient má SpO2 devadesát dva procent a tep sto dvacet za minutu."
Výstup: {{"polozky": {{"SpO2": 92, "Srdeční frekvence": 120}}}}

Uživatel: "Nasadil jsem kyslíkovou terapii, dva litry."
Výstup: {{"polozky": {{"Kyslíková terapie": "2 l/min"}}}}

Uživatel: "Tlak sto třicet na osmdesát."
Výstup: {{"polozky": {{"Krevní tlak": "130/80"}}}}
"""

# --- INICIALIZACE GROQ KLIENTA ---
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    logging.warning(f"Nepodařilo se inicializovat GROQ klienta. API klíč chybí? Chyba: {e}")
    groq_client = None

# --- HLAVNÍ TŘÍDA PRO ZPRACOVÁNÍ ŘEČI ---
class VoiceSpeechProcessor:
    def __init__(self):
        self.vosk_model = None
        self.recognizer = None
        self.p = None
        self.stream = None
        self.is_speaking = False
        self.speech_frames = deque()
        self.silence_counter = 0
        self.audio_buffer = deque(maxlen=int(RATE * (MIN_SPEECH_DURATION + SILENCE_DURATION) / CHUNK)) # Buffer pro zachycení začátku řeči
        if AI_PROVIDER == 'local':
            self.initialize_vosk()

    def initialize_vosk(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Vosk model nenalezen: {MODEL_PATH}")
            self.vosk_model = Model(MODEL_PATH)
            self.recognizer = KaldiRecognizer(self.vosk_model, RATE)
            self.recognizer.SetWords(True)
            logging.info("Vosk model úspěšně načten")
        except Exception as e:
            logging.error(f"Chyba při inicializaci Vosk: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': 'Chyba při inicializaci Vosk modelu.'})
            raise

    def initialize_audio(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK
            )
            logging.info("Audio stream inicializován")
        except Exception as e:
            logging.error(f"Chyba při inicializaci audio: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': 'Chyba audio zařízení.'})
            self.cleanup()
            raise

    def is_speech_detected(self, audio_data):
        rms = np.sqrt(np.mean(np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)**2))
        return rms > SILENCE_THRESHOLD

    def transcribe_audio_vosk(self, audio_frames):
        if not self.recognizer:
            logging.error("Vosk recognizer není inicializován.")
            return None
        
        try:
            self.recognizer.Reset()
            audio_data = b''.join(audio_frames)
            results = []
            
            for i in range(0, len(audio_data), CHUNK * 10):
                chunk = audio_data[i:i + CHUNK * 10]
                if self.recognizer.AcceptWaveform(chunk):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        results.append(result['text'])

            final_result = json.loads(self.recognizer.FinalResult())
            if final_result.get('text'):
                results.append(final_result['text'])

            full_text = ' '.join(results).strip()
            logging.info(f"Vosk přepis: '{full_text}'")
            return full_text if full_text else None
        except Exception as e:
            logging.error(f"Chyba při Vosk transkripci: {e}", exc_info=True)
            return None


    def transcribe_audio_groq(self, audio_frames):
        if not groq_client:
            logging.error("GROQ klient není k dispozici pro přepis.")
            socketio.emit('processing_error', {'message': 'GROQ klient není dostupný.'})
            return None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as tmp_wav_file:
                with wave.open(tmp_wav_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_frames))
                tmp_wav_path = tmp_wav_file.name

            with open(tmp_wav_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    response_format="text"
                )
            os.remove(tmp_wav_path)
            logging.info(f"GROQ přepis: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Chyba při GROQ transkripci: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': f'Chyba při GROQ transkripci: {e}'})
            return None

    def process_audio_chunk(self, data):
        self.audio_buffer.append(data)
        speech_detected = self.is_speech_detected(data)

        if not self.is_speaking and speech_detected:
            self.is_speaking = True
            self.silence_counter = 0
            self.speech_frames.clear()
            self.speech_frames.extend(self.audio_buffer)
            socketio.emit('speech_start')
            logging.info("Začátek řeči detekován")

        elif self.is_speaking:
            if speech_detected:
                self.silence_counter = 0
                self.speech_frames.append(data)
            else:
                self.silence_counter += 1
                self.speech_frames.append(data)
                required_silence_chunks = int(SILENCE_DURATION * RATE / CHUNK)
                if self.silence_counter > required_silence_chunks:
                    self.finalize_speech()

        if self.is_speaking and len(self.speech_frames) > (MAX_RECORDING_TIME * RATE / CHUNK):
            logging.warning("Dosažena maximální délka nahrávky, finalizuji.")
            self.finalize_speech()

    def finalize_speech(self):
        if not self.is_speaking or not self.speech_frames:
            return

        logging.info("Finalizuji řeč...")
        frames_to_process = list(self.speech_frames)
        
        self.is_speaking = False
        self.speech_frames.clear()
        self.silence_counter = 0
        
        socketio.emit('speech_end')

        def process_in_background():
            try:
                transcribed_text = None
                if AI_PROVIDER == 'local':
                    transcribed_text = self.transcribe_audio_vosk(frames_to_process)
                elif AI_PROVIDER == 'groq':
                    transcribed_text = self.transcribe_audio_groq(frames_to_process)

                if transcribed_text:
                    socketio.emit('transcription_result', {'text': transcribed_text})
                    process_with_llm(transcribed_text)
                else:
                    logging.warning("Přepis nevrátil žádný text.")
                    socketio.emit('transcription_result', {'text': ''})
                    socketio.emit('processing_error', {'message': 'Nepodařilo se rozpoznat žádný text.'})
            except Exception as e:
                logging.error(f"Chyba ve vláknu pro zpracování řeči: {e}", exc_info=True)
                socketio.emit('processing_error', {'message': 'Interní chyba serveru při zpracování.'})

        threading.Thread(target=process_in_background).start()

    def start_listening(self):
        global is_recording
        try:
            self.initialize_audio()
            logging.info("Zahajuji nahrávání...")
            while is_recording:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                if data and is_recording:
                    self.process_audio_chunk(data)
        except Exception as e:
            logging.error(f"Chyba v nahrávací smyčce: {e}", exc_info=True)
        finally:
            self.cleanup()

    def stop_listening(self):
        global is_recording
        is_recording = False
        self.finalize_speech()

    def cleanup(self):
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            self.stream = None
            self.p = None
            logging.info("Audio stream ukončen")
        except Exception as e:
            logging.error(f"Chyba při čištění audio zdrojů: {e}")

# --- ZPRACOVÁNÍ DAT POMOCÍ LLM ---
def get_data_from_ollama(text: str) -> dict | None:
    try:
        response = ollama.chat(
            model='gemma3:1b',
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            options={'response_format': {'type': 'json_object'}}
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        logging.error(f"Chyba při komunikaci s Ollama: {e}")
        return None

def get_data_from_groq(text: str) -> dict | None:
    if not groq_client:
        logging.error("GROQ klient není dostupný pro LLM zpracování.")
        socketio.emit('processing_error', {'message': 'GROQ klient není dostupný.'})
        return None
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            model="llama-3.1-8b-instant", 
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"Chyba při komunikaci s GROQ: {e}")
        return None

def process_with_llm(text: str):
    global df_numeric_state, df_text_state
    logging.info(f"Zpracovávám text: '{text}' pomocí {AI_PROVIDER}")
    
    extracted_data = None
    if AI_PROVIDER == 'local':
        extracted_data = get_data_from_ollama(text)
    elif AI_PROVIDER == 'groq' and groq_client:
        extracted_data = get_data_from_groq(text)

    if extracted_data and 'polozky' in extracted_data and extracted_data['polozky']:
        logging.info(f"LLM extrahoval data: {extracted_data}")
        timestamp = datetime.now().strftime('%H:%M:%S')

        last_numeric_col = df_numeric_state.columns[-1]
        last_text_col = df_text_state.columns[-1]
        
        df_numeric_state[timestamp] = df_numeric_state[last_numeric_col]
        df_text_state[timestamp] = df_text_state[last_text_col]
        
        items = extracted_data['polozky']
        for item_name, value in items.items():
            found_item = next((i for i in numeric_items + text_items if i.lower() == item_name.lower()), None)
            
            if not found_item:
                logging.warning(f"Položka '{item_name}' nebyla nalezena v definovaných seznamech.")
                continue

            if found_item in numeric_items:
                try:
                    if "/" in str(value):
                        df_numeric_state.loc[found_item, timestamp] = str(value)
                    else:
                        df_numeric_state.loc[found_item, timestamp] = float(value)
                except (ValueError, TypeError):
                    logging.warning(f"Nečíselná hodnota '{value}' pro číselnou položku '{found_item}'. Ukládám jako text.")
                    df_numeric_state.loc[found_item, timestamp] = str(value)
            
            elif found_item in text_items:
                df_text_state.loc[found_item, timestamp] = str(value)
        
        emit_table_update(extracted_data)
    else:
        logging.info("Žádná relevantní data k aktualizaci.")
        socketio.emit('processing_error', {'message': 'Nerozuměl jsem, žádná data k aktualizaci.'})

def emit_table_update(extracted_data=None):
    html_numeric = df_numeric_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    html_text = df_text_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    
    socketio.emit('table_update', {
        'numeric_table': html_numeric,
        'text_table': html_text,
        'extracted_data': extracted_data
    })
    logging.info("Tabulky aktualizovány a odeslány klientovi.")

# --- FLASK ROUTES & SOCKETIO EVENTS ---
@app.route('/')
def index():
    html_numeric = df_numeric_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    html_text = df_text_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    return render_template('index.html', numeric_table=html_numeric, text_table=html_text, provider=AI_PROVIDER)

@app.route('/process', methods=['POST'])
def process_text():
    user_text = request.form['text_input']
    if user_text:
        threading.Thread(target=process_with_llm, args=(user_text,)).start()
    return redirect(url_for('index'))

@socketio.on('set_ai_provider')
def set_ai_provider(data):
    global AI_PROVIDER
    new_provider = data.get('provider')
    if new_provider in ['local', 'groq']:
        AI_PROVIDER = new_provider
        logging.info(f"AI provider byl změněn na: {AI_PROVIDER}")
        emit('provider_changed', {'provider': AI_PROVIDER})

@socketio.on('start_recording')
def handle_start_recording():
    global speech_processor, is_recording
    with recording_lock:
        if not is_recording:
            is_recording = True
            speech_processor = VoiceSpeechProcessor()
            threading.Thread(target=speech_processor.start_listening, daemon=True).start()
            emit('recording_started')
            logging.info("Nahrávání spuštěno klientem.")

@socketio.on('stop_recording')
def handle_stop_recording():
    global speech_processor, is_recording
    with recording_lock:
        if is_recording and speech_processor:
            speech_processor.stop_listening() 
            is_recording = False
            emit('recording_stopped')
            logging.info("Nahrávání zastaveno klientem.")

def cleanup_app():
    global is_recording
    if is_recording:
        logging.info("Aplikace se ukončuje, zastavuji nahrávání.")
        is_recording = False

atexit.register(cleanup_app)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        logging.error(f"CHYBA: Vosk model nenalezen na cestě: {MODEL_PATH}")
    if not os.environ.get("GROQ_API_KEY"):
        logging.warning("Proměnná prostředí GROQ_API_KEY není nastavena. GROQ funkce nebudou dostupné.")

    socketio.run(app, debug=False, host='0.0.0.0', port=5050)