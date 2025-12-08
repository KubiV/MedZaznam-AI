import ollama
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
import json
import pandas as pd
import logging
import os
import threading
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
from datetime import datetime
from collections import deque
import atexit
import tempfile
import wave
import shutil
import glob
import time  # <--- PRIDANO pro throttling vizualizace

# Import pro konverzi do MP3
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("UPOZORNĚNÍ: Knihovna 'pydub' chybí. Nahrávky zůstanou ve formátu WAV. (pip install pydub)")

# --- NASTAVENÍ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_BASE_DIR = os.path.join(BASE_DIR, 'sessions') # Nová hlavní složka pro seance
# MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-cs-0.4-rhasspy") 
MODEL_PATH = "/Users/jakubvavra/Documents/GitHub/Automonitoring-with-AI/tests/vosk/vosk-model-small-cs-0.4-rhasspy"

# --- AUDIO PARAMETRY ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 150
MIN_SPEECH_DURATION = 0.5  # V sekundách
SILENCE_DURATION = 1.5     # V sekundách
MAX_RECORDING_TIME = 10    # V sekundách

# --- NAČTENÍ ENV PROMĚNNÝCH ---
load_dotenv()

# --- INICIALIZACE APLIKACE ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- DOSTUPNÉ MODELY A NASTAVENÍ (NOVÉ) ---
AVAILABLE_MODELS = {
    "stt": [
        "vosk-local",              # Lokální Vosk
        "whisper-large-v3-turbo",  # Groq Whisper
        "whisper-large-v3"         # Groq Whisper
    ],
    "llm": [
        "gemma3:4b",               # Lokální Ollama
        "gemma3:1b",               # Lokální Ollama
        "deepseek-r1:1.5b",        # Lokální Ollama
        "llama-3.3-70b-versatile", # Groq
        "llama-3.1-8b-instant",    # Groq
        "openai/gpt-oss-120b",     # Groq
        "gemini-1.5-flash",        # Google
        "gemini-2.5-flash-lite",   # Google
        "gemini-2.5-flash"         # Google
    ]
}

CURRENT_SETTINGS = {
    "stt_model": "vosk-local",
    "llm_model": "gemma3:4b",
    "temperature": 0.1
}

# --- GLOBÁLNÍ PROMĚNNÉ A ZÁMKY ---

# -- Nastavení nahrávání --
SAVE_CONTINUOUS_AUDIO = False # Výchozí stav ukládání audia
current_session_dir = None    # Cesta k aktuální složce seance
SELECTED_DEVICE_INDEX = None  # Index vybraného mikrofonu (None = default)

speech_processor = None
is_recording = False
recording_lock = threading.Lock()

# --- DEFINICE KATEGORIÍ A SOUBORŮ ---
CATEGORY_FILES = {
    "DrABCDE": "tables/DrABCDE.csv",
    "Medication": "tables/Medication.csv",
    "Interventions": "tables/Interventions.csv",
    "Physical Examination": "tables/Physical Examination.csv",
    "History": "tables/History.csv",
    "SBAR": "tables/SBAR.csv",
    "Other": "tables/Other.csv" 
}

# Globální úložiště pro tabulky a mapování
data_tables = {}     # { "DrABCDE": DataFrame, "Medication": DataFrame, ... }
item_mapping = {}    # { "Položka": "DrABCDE", ... } slouží k rychlému nalezení, do jaké tabulky položka patří
all_known_items = [] # Seznam všech položek pro LLM prompt

# --- NOVÉ GLOBÁLNÍ PROMĚNNÉ PRO DASHBOARD (TF, TK, atd.) ---

# 1. KROK - SYNONYMA: (ROZŠÍŘENO O MEDIKACI DLE VERCEL)
ITEM_SYNONYMS = {
    # Všechny varianty tepu -> "Srdeční frekvence" (aby to sedělo do DrABCDE)
    "tepová frekvence": "Srdeční frekvence",
    "typová frekvence": "Srdeční frekvence",
    "tep": "Srdeční frekvence",
    "puls": "Srdeční frekvence",
    "sf": "Srdeční frekvence",
    "tf": "Srdeční frekvence",
    "srdeční frekvence (sf)": "Srdeční frekvence",
    
    # Tlak
    "tlak": "Krevní tlak",
    "tk": "Krevní tlak",
    "systolický tlak": "Krevní tlak",
    
    # Saturace
    "saturace": "SpO2",
    "sat": "SpO2",
    "spo2": "SpO2",
    
    # Dech
    "dech": "Dechová frekvence",
    "df": "Dechová frekvence",
    
    # Vědomí
    "vědomí": "AVPU",
    "gcs": "glasgow coma scale",
    "glasgow coma scale": "gcs",
    "avpu": "AVPU",
    
    # CRT
    "CRT": "kapilární návrat",
    "crt": "kapilární návrat",

    # --- NOVÉ: Synonyma pro Léčiva (Medication.csv) ---
    "adrenalin": "Adrenalin",
    "amiodaron": "Amiodaron",
    "atropin": "Atropin",
    "noradrenalin": "Noradrenalin",
    "adenosin": "Adenosin",
    "midazolam": "Midazolam",
    "fentanyl": "Fentanyl",
    "morfin": "Morfin",
    "ketamin": "Ketamin",
    "propofol": "Propofol",
    "sufentanil": "Sufentanil",
    "rokuronium": "Rokuronium",
    "sukcinylcholin": "Sukcinylcholin",
    
    # Složené názvy
    "aspirin": "Aspirin (ASA)", 
    "asa": "Aspirin (ASA)",
    
    "heparin": "Heparin",
    
    "nitroglycerin": "Nitroglycerin (Isoket)", 
    "isoket": "Nitroglycerin (Isoket)",
    
    "furosemid": "Furosemid",
    
    "salbutamol": "Salbutamol (Ventolin)", 
    "ventolin": "Salbutamol (Ventolin)",
    
    "urapidil": "Urapidil (Ebrantil)", 
    "ebrantil": "Urapidil (Ebrantil)",
    
    "exacyl": "Exacyl (Kys. tranexamová)", 
    "kys. tranexamová": "Exacyl (Kys. tranexamová)",
    "kyselina tranexamová": "Exacyl (Kys. tranexamová)",
    
    "magnesium sulfát": "Magnesium sulfát",
    "magnesium": "Magnesium sulfát",
    
    "naloxon": "Naloxon",
    
    "flumazenil": "Flumazenil (Anexate)", 
    "anexate": "Flumazenil (Anexate)",
    
    "glukóza": "Glukóza (G40%)", 
    "g40%": "Glukóza (G40%)",
    "g40": "Glukóza (G40%)",
    
    "paracetamol": "Paracetamol",
    "ondansetron": "Ondansetron",
    "ceftriaxon": "Ceftriaxon"
}

# 2. KROK - MAPPING NA DISPLEJ: Oficiální název z CSV -> Zkratka na dashboardu
VITALS_MAPPING = {
    "srdeční frekvence": "TF",
    "krevní tlak": "TK",
    "dechová frekvence": "DF",
    "spo2": "SpO2",
    "kapilární návrat": "CRT",
    "avpu": "AVPU"
}

# Výchozí stav dashboardu
current_vitals = {
    "TF": "--",
    "TK": "--/--",
    "DF": "--",
    "SpO2": "--",
    "CRT": "--",
    "AVPU": "--"
}

# Historie posledních 5 změn: seznam slovníků {time, item, value}
recent_updates = deque(maxlen=5)

# --- SPRÁVA LOGOVÁNÍ A ADRESÁŘŮ ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Toto zajistí výpis do konzole vždy
    ]
)

def setup_logging(session_dir):
    """
    UPRAVENO: Přidá FileHandler pro aktuální session, ale zachová StreamHandler (konzoli).
    """
    log_file = os.path.join(session_dir, 'session_log.log')
    logger = logging.getLogger()
    
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info(f"--- LOGOVÁNÍ NASTAVENO DO: {log_file} ---")
    log_startup_parameters()

def log_startup_parameters():
    logging.info("="*60)
    logging.info("INICIALIZACE APLIKACE - PARAMETRY SPUŠTĚNÍ")
    logging.info("="*60)
    
    logging.info("--- AUDIO PARAMETRY ---")
    logging.info(f"RATE: {RATE} Hz, CHUNK: {CHUNK}, CHANNELS: {CHANNELS}")
    
    logging.info("--- NASTAVENÍ AI MODELŮ ---")
    logging.info(f"STT Model: {CURRENT_SETTINGS['stt_model']}")
    logging.info(f"LLM Model: {CURRENT_SETTINGS['llm_model']}")
    
    logging.info("--- CESTY ---")
    logging.info(f"BASE_DIR: {BASE_DIR}")
    
    logging.info("--- API KLÍČE ---")
    logging.info(f"GROQ_API_KEY nastaveno: {bool(os.environ.get('GROQ_API_KEY'))}")
    logging.info(f"GOOGLE_API_KEY nastaveno: {bool(os.environ.get('GOOGLE_API_KEY'))}")
    
    logging.info("="*60)

def get_or_create_session_dir(force_new=False):
    global current_session_dir
    
    if not os.path.exists(SESSIONS_BASE_DIR):
        os.makedirs(SESSIONS_BASE_DIR)

    if force_new:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"Session_{timestamp}"
        current_session_dir = os.path.join(SESSIONS_BASE_DIR, session_name)
        os.makedirs(current_session_dir, exist_ok=True)
        setup_logging(current_session_dir) 
        return current_session_dir
    
    if current_session_dir and os.path.exists(current_session_dir):
        return current_session_dir
    
    subdirs = glob.glob(os.path.join(SESSIONS_BASE_DIR, 'Session_*'))
    if subdirs:
        latest_session = max(subdirs, key=os.path.getctime)
        current_session_dir = latest_session
        setup_logging(current_session_dir)
        return current_session_dir
    else:
        return get_or_create_session_dir(force_new=True)

# --- NAČTENÍ A PŘÍPRAVA DAT ---
def initialize_data_structures():
    global data_tables, item_mapping, all_known_items
    
    data_tables = {}
    item_mapping = {}
    all_known_items = []
    
    for category, filename in CATEGORY_FILES.items():
        csv_path = os.path.join(BASE_DIR, filename)
        
        try:
            df = pd.read_csv(csv_path, delimiter=';')
            first_col_name = df.columns[0]
            df.set_index(first_col_name, inplace=True)
            df.index.name = "Položka"
            df['Aktuální stav'] = ""
            clean_df = df[['Aktuální stav']].copy()
            data_tables[category] = clean_df
            
            for item in clean_df.index:
                item_str = str(item).strip()
                if item_str:
                    item_mapping[item_str.lower()] = category
                    all_known_items.append(item_str)
                    
            logging.info(f"Načtena kategorie {category} ze souboru {filename} ({len(clean_df)} položek).")

        except FileNotFoundError:
            if category == "Other":
                logging.info(f"Info: Vytvářím prázdnou tabulku pro kategorii {category}.")
            else:
                logging.error(f"POZOR: Soubor {filename} pro kategorii {category} nebyl nalezen! Vytvářím prázdnou tabulku.")
            
            data_tables[category] = pd.DataFrame(columns=['Aktuální stav'])
            data_tables[category].index.name = "Položka"
        except Exception as e:
            logging.error(f"Chyba při načítání {filename}: {e}")

initialize_data_structures()
get_or_create_session_dir(force_new=False)


# --- LLM PROMPTY ---
items_list_str = ', '.join(f'"{item}"' for item in all_known_items)

system_prompt = f"""
Jsi expert na extrakci lékařských dat z mluveného slova v reálném čase.
Tvým úkolem je naslouchat hlášením lékaře/záchranáře a strukturovat je do JSON formátu podle definovaných položek.
Údaje si NEVYMYŠLEJ, pokud nejsou explicitně uvedeny v textu, ponech je prázdné.
Pracuj pouze s textem v českém jazyce.

Pravidla:
1.  Výstup musí být VŽDY a POUZE platný JSON objekt.
2.  JSON obsahuje klíč "polozky", což je slovník.
3.  Klíče ve slovníku "polozky" musí odpovídat názvům z následujícího seznamu (nebo jejich blízkým synonymům, které normalizuješ na tento seznam):
    [{items_list_str}]
4.  Normalizuj synonyma na oficiální názvy ze seznamu (např. "tep" -> "Srdeční frekvence", "saturace" -> "SpO2", "Isoket" -> "Nitroglycerin (Isoket)").
5.  Hodnoty:
    - Čísla extrahuj jako čísla nebo formátovaný string (např. "120/80").
    - Textové popisy extrahuj stručně a jasně.
6.  Pokud text neobsahuje žádné relevantní informace, vrať prázdný slovník `{{"polozky": {{}}}}`.

Příklady:
Uživatel: "Pacient má saturaci 92 a tlak 130 na 80. Podán Isoket jeden vstřik."
Výstup: {{"polozky": {{"SpO2": "92", "Krevní tlak": "130/80", "Nitroglycerin (Isoket)": "1 vstřik"}}}}

Uživatel: "Podávám 1 miligram adrenalinu intravenózně."
Výstup: {{"polozky": {{"Adrenalin": "1 mg i.v."}}}}

Uživatel: "Pacient je mírně dušný, bez alergie."
Výstup: {{"polozky": {{"Dušnost": "Mírná", "Alergická anamnéza (AA)": "Neguje"}}}}
"""

# --- INICIALIZACE API KLIENTŮ ---
groq_client = None
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    logging.warning(f"Nepodařilo se inicializovat GROQ klienta: {e}")

# Google AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logging.warning("Chybí GOOGLE_API_KEY, modely Gemini nebudou fungovat.")


# --- POMOCNÉ AUDIO FUNKCE ---
def get_audio_input_devices():
    p = pyaudio.PyAudio()
    devices = []
    try:
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                devices.append({'index': i, 'name': name})
    finally:
        p.terminate()
    return devices

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
        self.audio_buffer = deque(maxlen=int(RATE * (MIN_SPEECH_DURATION + SILENCE_DURATION) / CHUNK))
        
        self.continuous_wave_file = None
        self.continuous_file_path_wav = None
        self.last_emit_time = 0 # Pro vizualizaci na klientovi
        
        # Inicializace modelu pouze pokud je vybrán lokální Vosk
        if CURRENT_SETTINGS['stt_model'] == 'vosk-local':
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
            kwargs = {
                'format': pyaudio.paInt16,
                'channels': CHANNELS,
                'rate': RATE,
                'input': True,
                'frames_per_buffer': CHUNK
            }
            if SELECTED_DEVICE_INDEX is not None:
                kwargs['input_device_index'] = int(SELECTED_DEVICE_INDEX)
                device_info = self.p.get_device_info_by_index(int(SELECTED_DEVICE_INDEX))
                device_name = device_info.get('name', 'Unknown')
                logging.info(f"Otevírám audio stream na zařízení index: {SELECTED_DEVICE_INDEX} ({device_name})")
            else:
                default_device = self.p.get_default_input_device_info()
                default_name = default_device.get('name', 'Default Device')
                logging.info(f"Otevírám audio stream na výchozím zařízení: {default_name}")

            self.stream = self.p.open(**kwargs)
            logging.info("Audio stream inicializován")
            
            if SAVE_CONTINUOUS_AUDIO:
                session_dir = get_or_create_session_dir(force_new=False)
                timestamp = datetime.now().strftime('%H%M%S')
                self.continuous_file_path_wav = os.path.join(session_dir, f'recording_{timestamp}.wav')
                
                self.continuous_wave_file = wave.open(self.continuous_file_path_wav, 'wb')
                self.continuous_wave_file.setnchannels(CHANNELS)
                self.continuous_wave_file.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                self.continuous_wave_file.setframerate(RATE)
                logging.info(f"Zahájeno nahrávání do souboru: {self.continuous_file_path_wav}")

        except Exception as e:
            logging.error(f"Chyba při inicializaci audio: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': f'Chyba audio zařízení: {str(e)}'})
            self.cleanup()
            raise

    def is_speech_detected(self, audio_data):
        # Bezpečný výpočet RMS s přetypováním na float, aby nedošlo k přetečení int16
        data_int16 = np.frombuffer(audio_data, dtype=np.int16)
        data_float = data_int16.astype(np.float64)
        rms = np.sqrt(np.mean(data_float**2))
        return rms, rms > SILENCE_THRESHOLD

    def transcribe_audio_vosk(self, audio_frames):
        if not self.recognizer:
            logging.error("Vosk recognizer není inicializován.")
            return None
        
        try:
            self.recognizer.Reset()
            audio_data = b''.join(audio_frames)
            results = []
            
            # Zpracování po částech
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
            logging.info(f"[VOSK PŘEPIS] {full_text}")
            return full_text if full_text else None
        except Exception as e:
            logging.error(f"Chyba při Vosk transkripci: {e}", exc_info=True)
            return None

    def transcribe_audio_groq(self, audio_frames):
        if not groq_client:
            logging.error("GROQ klient není k dispozici.")
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

            selected_stt_model = CURRENT_SETTINGS.get('stt_model', 'whisper-large-v3-turbo')
            # Pokud je z nějakého důvodu nastaveno 'vosk-local', ale volá se tato metoda, použijeme fallback
            if selected_stt_model == 'vosk-local': 
                selected_stt_model = 'whisper-large-v3-turbo'

            with open(tmp_wav_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    model=selected_stt_model,
                    file=audio_file,
                    response_format="text",
                    language="cs"
                )
            os.remove(tmp_wav_path)
            logging.info(f"[GROQ ({selected_stt_model}) PŘEPIS] {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Chyba při GROQ transkripci: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': f'Chyba při GROQ transkripci: {e}'})
            return None

    def process_audio_chunk(self, data):
        if self.continuous_wave_file:
            self.continuous_wave_file.writeframes(data)

        self.audio_buffer.append(data)
        
        # Získání RMS a rozhodnutí o řeči
        rms, speech_detected = self.is_speech_detected(data)

        # --- VIZUALIZACE PRO KLIENTA ---
        # Posíláme data zpět na klienta, aby viděl "serverový" pohled na mikrofon
        current_time = time.time()
        # Omezíme frekvenci odesílání na cca 20x za sekundu, aby se nezahltila síť
        if current_time - self.last_emit_time > 0.05:
            socketio.emit('audio_level', {
                'rms': float(rms),
                'threshold': SILENCE_THRESHOLD,
                'is_speaking': self.is_speaking
            })
            self.last_emit_time = current_time

        if not self.is_speaking and speech_detected:
            self.is_speaking = True
            self.silence_counter = 0
            self.speech_frames.clear()
            self.speech_frames.extend(self.audio_buffer)
            socketio.emit('speech_start')

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
            logging.warning("Dosažena maximální délka segmentu, finalizuji.")
            self.finalize_speech()

    def finalize_speech(self):
        if not self.is_speaking or not self.speech_frames:
            return

        logging.info("Finalizuji segment řeči...")
        frames_to_process = list(self.speech_frames)
        
        self.is_speaking = False
        self.speech_frames.clear()
        self.silence_counter = 0
        
        socketio.emit('speech_end')

        def process_in_background():
            try:
                transcribed_text = None
                stt_model = CURRENT_SETTINGS['stt_model']
                
                # Rozhodování podle nastaveného modelu
                if stt_model == 'vosk-local':
                    transcribed_text = self.transcribe_audio_vosk(frames_to_process)
                else:
                    # Ostatní modely (Whisper) jdou přes Groq
                    transcribed_text = self.transcribe_audio_groq(frames_to_process)

                if transcribed_text:
                    socketio.emit('transcription_result', {'text': transcribed_text})
                    process_with_llm(transcribed_text)
                else:
                    logging.warning("Přepis nevrátil žádný text.")
                    socketio.emit('transcription_result', {'text': ''})
            except Exception as e:
                logging.error(f"Chyba ve vláknu zpracování: {e}", exc_info=True)
                socketio.emit('processing_error', {'message': 'Interní chyba serveru.'})

        threading.Thread(target=process_in_background).start()

    def start_listening(self):
        global is_recording
        try:
            self.initialize_audio()
            logging.info("Zahajuji nahrávací smyčku...")
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
        
    def convert_wav_to_mp3(self, wav_path):
        if not HAS_PYDUB:
            return
        
        mp3_path = wav_path.replace('.wav', '.mp3')
        try:
            logging.info(f"Konvertuji záznam do MP3: {mp3_path}")
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3")
            logging.info("Konverze dokončena.")
            
            os.remove(wav_path)
        except Exception as e:
            logging.error(f"Chyba při konverzi do MP3: {e}")

    def cleanup(self):
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            
            if self.continuous_wave_file:
                self.continuous_wave_file.close()
                self.continuous_wave_file = None
                logging.info("Soubor s nahrávkou uzavřen.")
                
                if self.continuous_file_path_wav and os.path.exists(self.continuous_file_path_wav):
                    threading.Thread(target=self.convert_wav_to_mp3, args=(self.continuous_file_path_wav,)).start()

            self.stream = None
            self.p = None
            logging.info("Audio resources uvolněny")
        except Exception as e:
            logging.error(f"Chyba při čištění audio zdrojů: {e}")

# --- ZPRACOVÁNÍ DAT POMOCÍ LLM ---
def get_data_from_ollama(text: str, model_name: str) -> dict | None:
    try:
        logging.info(f"Odesílám do Ollama ({model_name})...")
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            options={'temperature': CURRENT_SETTINGS['temperature'], 'response_format': {'type': 'json_object'}}
        )
        
        content = response['message']['content']
        logging.info(f"Ollama odpověď (raw): {content}")
        
        if not content:
            logging.error("Ollama vrátila prázdnou odpověď.")
            return None

        # --- OPRAVA: Odstranění Markdown značek ---
        clean_content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(clean_content)

    except json.JSONDecodeError as e:
        logging.error(f"Ollama nevrátila validní JSON. Chyba: {e}")
        # Pro jistotu vypíšeme, co jsme se snažili parsovat po očištění
        logging.error(f"Data po očištění: {clean_content if 'clean_content' in locals() else 'N/A'}")
        return None
    except Exception as e:
        logging.error(f"Kritická chyba při komunikaci s Ollama: {e}")
        return None
    try:
        logging.info(f"Odesílám do Ollama ({model_name})...")
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            options={'temperature': CURRENT_SETTINGS['temperature'], 'response_format': {'type': 'json_object'}}
        )
        
        content = response['message']['content']
        # Logování obsahu pro kontrolu
        logging.info(f"Ollama odpověď (raw): {content}")
        
        if not content:
            logging.error("Ollama vrátila prázdnou odpověď.")
            return None
            
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"Ollama nevrátila validní JSON. Chyba: {e}")
        logging.error(f"Přijatá data: {content if 'content' in locals() else 'Žádná data'}")
        return None
    except Exception as e:
        logging.error(f"Kritická chyba při komunikaci s Ollama: {e}")
        return None

def get_data_from_groq(text: str, model_name: str) -> dict | None:
    if not groq_client:
        logging.error("GROQ klient není dostupný.")
        return None
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            model=model_name,
            temperature=CURRENT_SETTINGS['temperature'],
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"Chyba při komunikaci s GROQ: {e}")
        return None

def get_data_from_google(text: str, model_name: str) -> dict | None:
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        response = model.generate_content(
            text, 
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Chyba při komunikaci s Google AI: {e}")
        return None

def save_to_csv():
    try:
        session_dir = get_or_create_session_dir(force_new=False)
        for category, df in data_tables.items():
            safe_cat_name = category.replace(" ", "_")
            csv_path = os.path.join(session_dir, f'vysledny_stav_{safe_cat_name}.csv')
            df.to_csv(csv_path)
        logging.info(f"Všechny tabulky aktualizovány v: {session_dir}")
    except Exception as e:
        logging.error(f"Chyba při ukládání CSV: {e}")

def process_with_llm(text: str):
    # DŮLEŽITÉ: global musí být na úplně prvním řádku po definici funkce
    global data_tables, item_mapping, current_vitals, recent_updates
    
    get_or_create_session_dir(force_new=False)
    
    llm_model = CURRENT_SETTINGS['llm_model']
    logging.info(f"Zpracovávám text: '{text}' pomocí {llm_model}")
    
    extracted_data = None
    
    # --- OPRAVENÁ ROZHODOVACÍ LOGIKA (zahrnuje fix pro openai/gpt-oss) ---
    model_lower = llm_model.lower()
    
    # 1. Google Gemini
    if "gemini" in model_lower:
        extracted_data = get_data_from_google(text, llm_model)
        
    # 2. Groq (Llama, Mixtral, GPT, OpenAI atd.)
    elif "llama" in model_lower or "mixtral" in model_lower or "gpt" in model_lower or "openai" in model_lower:
        extracted_data = get_data_from_groq(text, llm_model)
        
    # 3. Fallback na lokální Ollama
    else:
        extracted_data = get_data_from_ollama(text, llm_model)

    # Zpracování výsledků
    if extracted_data and 'polozky' in extracted_data and extracted_data['polozky']:
        logging.info(f"LLM extrahoval data: {extracted_data}")
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        items = extracted_data['polozky']
        updated_categories = set()

        for item_name, value in items.items():
            item_lower = item_name.lower()

            # --- SYNONYM CHECK ---
            if item_lower in ITEM_SYNONYMS:
                original_name = item_name
                item_name = ITEM_SYNONYMS[item_lower]
                item_lower = item_name.lower()
                logging.info(f"Synonymum nahrazeno: {original_name} -> {item_name}")

            # --- HISTORIE ---
            recent_updates.appendleft(f"{timestamp} - {item_name} - {value}")

            # --- DASHBOARD UPDATE ---
            vital_key = VITALS_MAPPING.get(item_lower)
            if vital_key:
                current_vitals[vital_key] = str(value)

            # --- LOGIKA TABULEK ---
            category = item_mapping.get(item_lower)
            
            if not category:
                for known_item, cat in item_mapping.items():
                    if known_item == item_lower:
                        category = cat
                        # Najdeme přesný název v indexu (kvůli velikosti písmen)
                        real_item_name_list = list(data_tables[cat].index[data_tables[cat].index.str.lower() == known_item])
                        if real_item_name_list:
                            item_name = real_item_name_list[0] 
                        break
            
            # --- FALLBACK DO "OTHER" ---
            if not category:
                logging.warning(f"Položka '{item_name}' nebyla nalezena. Přidávám ji do 'Other'.")
                category = "Other"
                if category in data_tables:
                    df = data_tables[category]
                    if item_name not in df.index:
                        new_row_data = {col: "" for col in df.columns}
                        df.loc[item_name] = new_row_data
                        item_mapping[item_lower] = category
                        all_known_items.append(item_name)
            
            # Zápis do tabulky
            if category and category in data_tables:
                df = data_tables[category]
                if timestamp not in df.columns:
                    df[timestamp] = ""
                
                if item_name in df.index:
                    df.loc[item_name, timestamp] = str(value)
                    updated_categories.add(category)
                else:
                    matching_index = df.index[df.index.str.lower() == item_lower]
                    if not matching_index.empty:
                        real_index = matching_index[0]
                        df.loc[real_index, timestamp] = str(value)
                        updated_categories.add(category)

        emit_table_update(extracted_data)
        save_to_csv() 
        
    else:
        logging.info("Žádná relevantní data k aktualizaci nebo chyba LLM.")
        socketio.emit('processing_error', {'message': 'Nerozuměl jsem nebo LLM selhalo.'})

def generate_html_tables():
    """Generuje HTML pro všech 7 tabulek."""
    tables_html = {}
    for category, df in data_tables.items():
        tables_html[category] = df.fillna('').to_html(classes="table table-striped table-hover table-sm", border=0)
    return tables_html

def emit_table_update(extracted_data=None):
    tables_html = generate_html_tables()
    updates_list = list(recent_updates)
    
    socketio.emit('table_update', {
        'tables': tables_html,
        'extracted_data': extracted_data,
        'current_vitals': current_vitals,
        'recent_updates': updates_list
    })
    logging.info("Tabulky a dashboard odeslány klientovi.")

# --- FLASK ROUTES & SOCKETIO EVENTS ---
@app.route('/')
def index():
    tables_html = generate_html_tables()
    return render_template('index.html', 
                           tables=tables_html,
                           available_models=AVAILABLE_MODELS,
                           current_settings=CURRENT_SETTINGS,
                           save_audio=SAVE_CONTINUOUS_AUDIO,
                           current_vitals=current_vitals,
                           recent_updates=list(recent_updates))

@app.route('/process', methods=['POST'])
def process_text():
    user_text = request.form['text_input']
    if user_text:
        logging.info(f"[MANUÁLNÍ VSTUP] {user_text}")
        threading.Thread(target=process_with_llm, args=(user_text,)).start()
    return redirect(url_for('index'))

@app.route('/sw.js')
def service_worker():
    response = send_from_directory('static', 'sw.js')
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@socketio.on('update_settings')
def handle_update_settings(data):
    """
    Nový handler pro nastavení modelu.
    Očekává: { 'stt_model': '...', 'llm_model': '...' }
    """
    global CURRENT_SETTINGS
    stt = data.get('stt_model')
    llm = data.get('llm_model')
    
    if stt in AVAILABLE_MODELS['stt']:
        CURRENT_SETTINGS['stt_model'] = stt
    if llm in AVAILABLE_MODELS['llm']:
        CURRENT_SETTINGS['llm_model'] = llm
        
    logging.info(f"Nastavení aktualizováno: {CURRENT_SETTINGS}")
    emit('settings_updated', CURRENT_SETTINGS)

@socketio.on('toggle_audio_save')
def toggle_audio_save(data):
    global SAVE_CONTINUOUS_AUDIO
    SAVE_CONTINUOUS_AUDIO = data.get('save', False)
    state_text = "ZAPNUTO" if SAVE_CONTINUOUS_AUDIO else "VYPNUTO"
    logging.info(f"Ukládání audia: {state_text}")
    emit('audio_save_changed', {'save': SAVE_CONTINUOUS_AUDIO})

@socketio.on('get_devices')
def handle_get_devices():
    devices = get_audio_input_devices()
    emit('devices_list', devices)

@socketio.on('set_device')
def handle_set_device(data):
    global SELECTED_DEVICE_INDEX
    device_index = data.get('device_index')
    
    if device_index == 'default':
        SELECTED_DEVICE_INDEX = None
    else:
        try:
            SELECTED_DEVICE_INDEX = int(device_index)
        except (ValueError, TypeError):
            SELECTED_DEVICE_INDEX = None
    
    logging.info(f"Vybrán mikrofon index: {SELECTED_DEVICE_INDEX}")

@socketio.on('reset_session')
def handle_reset_session():
    global current_vitals, recent_updates
    
    logging.info("Požadavek na reset session.")
    new_dir = get_or_create_session_dir(force_new=True)
    
    initialize_data_structures()
    
    current_vitals = {
        "TF": "--", "TK": "--/--", "DF": "--", "SpO2": "--", "CRT": "--", "AVPU": "--"
    }
    recent_updates.clear()
    
    emit_table_update(extracted_data={})
    logging.info(f"Session resetována. Nová složka: {os.path.basename(new_dir)}")
    emit('session_reset_complete', {'message': f'Nová seance zahájena: {os.path.basename(new_dir)}'})


@socketio.on('start_recording')
def handle_start_recording():
    global speech_processor, is_recording
    with recording_lock:
        if not is_recording:
            get_or_create_session_dir(force_new=False)
            
            is_recording = True
            # Inicializace procesoru se aktuálním nastavením (načte Vosk jen pokud je vybrán)
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
        logging.info("Ukončuji aplikaci, zastavuji nahrávání.")
        is_recording = False

atexit.register(cleanup_app)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        logging.warning(f"VAROVÁNÍ: Vosk model nenalezen: {MODEL_PATH} (Lokální STT nebude fungovat)")
    if not os.environ.get("GROQ_API_KEY"):
        logging.warning("Chybí GROQ_API_KEY.")

    socketio.run(app, debug=False, host='0.0.0.0', port=5050)