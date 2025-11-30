import ollama
from groq import Groq
from dotenv import load_dotenv
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
import shutil
import glob

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
CSV_DATA_PATH = os.path.join(BASE_DIR, 'Hodnoty MED.csv')

# --- AUDIO PARAMETRY ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 150
MIN_SPEECH_DURATION = 0.5  # V sekundách
SILENCE_DURATION = 1.5     # V sekundách
MAX_RECORDING_TIME = 10    # V sekundách

# --- INICIALIZACE APLIKACE ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- GLOBÁLNÍ PROMĚNNÉ A ZÁMKY ---
# Nastavení AI Providera
AI_PROVIDER = 'local'  # Výchozí hodnota: 'local' nebo 'groq'

# -- Nastavení pro LOKÁLNÍ OLLAMA modely --
LOCAL_MODEL_NAME = "gemma3" # gemma3:1b, gemma3:4b, deepseek-r1:1.5b
LOCAL_TEMPERATURE = 0.1     # Nižší teplota = přesnější extrakce dat, méně "kreativity"

# -- Nastavení pro GROQ modely --
GROQ_MODEL_NAME = "llama-3.1-8b-instant" # llama-3.3-70b-versatile, llama-3.1-8b-instant
GROQ_TEMPERATURE = 0.2
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3" # whisper-large-v3-turbo, whisper-large-v3

# -- Nastavení nahrávání --
SAVE_CONTINUOUS_AUDIO = False # Výchozí stav ukládání audia
current_session_dir = None    # Cesta k aktuální složce seance
SELECTED_DEVICE_INDEX = None  # Index vybraného mikrofonu (None = default)

speech_processor = None
is_recording = False
recording_lock = threading.Lock()


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
    Tím se docílí toho, že logging.info jde tam i tam.
    """
    log_file = os.path.join(session_dir, 'session_log.log')
    logger = logging.getLogger()
    
    # Odstraníme POUZE staré FileHandlery (aby se nepsalo do starých session souborů)
    # StreamHandler pro konzoli ponecháme.
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            
    # Přidáme nový file handler pro aktuální složku
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.info(f"--- LOGOVÁNÍ NASTAVENO DO: {log_file} ---")
    log_startup_parameters()

def log_startup_parameters():
    """
    Zaloguje všechny důležité parametry při startu aplikace.
    """
    logging.info("="*60)
    logging.info("INICIALIZACE APLIKACE - PARAMETRY SPUŠTĚNÍ")
    logging.info("="*60)
    
    # Audio parametry
    logging.info("--- AUDIO PARAMETRY ---")
    logging.info(f"RATE (vzorkovací frekvence): {RATE} Hz")
    logging.info(f"CHUNK (velikost bufferu): {CHUNK} byte")
    logging.info(f"CHANNELS (počet kanálů): {CHANNELS}")
    logging.info(f"SILENCE_THRESHOLD (práh ticha): {SILENCE_THRESHOLD}")
    logging.info(f"MIN_SPEECH_DURATION (minimální trvání řeči): {MIN_SPEECH_DURATION} s")
    logging.info(f"SILENCE_DURATION (trvání ticha pro ukončení): {SILENCE_DURATION} s")
    logging.info(f"MAX_RECORDING_TIME (maximální čas záznamu): {MAX_RECORDING_TIME} s")
    
    # AI Provider
    logging.info("--- AI POSKYTOVATEL ---")
    logging.info(f"AI_PROVIDER: {AI_PROVIDER}")
    
    # Lokální Ollama nastavení
    logging.info("--- LOKÁLNÍ OLLAMA NASTAVENÍ ---")
    logging.info(f"LOCAL_MODEL_NAME: {LOCAL_MODEL_NAME}")
    logging.info(f"LOCAL_TEMPERATURE: {LOCAL_TEMPERATURE}")
    
    # GROQ nastavení
    logging.info("--- GROQ NASTAVENÍ ---")
    logging.info(f"GROQ_MODEL_NAME: {GROQ_MODEL_NAME}")
    logging.info(f"GROQ_TEMPERATURE: {GROQ_TEMPERATURE}")
    logging.info(f"GROQ_TRANSCRIPTION_MODEL: {GROQ_TRANSCRIPTION_MODEL}")
    
    # Audio ukládání
    logging.info("--- NAHRÁVÁNÍ AUDIA ---")
    logging.info(f"SAVE_CONTINUOUS_AUDIO: {SAVE_CONTINUOUS_AUDIO}")
    
    # Cesty
    logging.info("--- CESTY A KONFIGURACE ---")
    logging.info(f"BASE_DIR: {BASE_DIR}")
    logging.info(f"SESSIONS_BASE_DIR: {SESSIONS_BASE_DIR}")
    logging.info(f"MODEL_PATH: {MODEL_PATH}")
    logging.info(f"CSV_DATA_PATH: {CSV_DATA_PATH}")
    logging.info(f"Model dostupný: {os.path.exists(MODEL_PATH)}")
    logging.info(f"CSV dostupný: {os.path.exists(CSV_DATA_PATH)}")
    
    # Mikrofon
    logging.info("--- NASTAVENÍ MIKROFONU ---")
    if SELECTED_DEVICE_INDEX is not None:
        logging.info(f"SELECTED_DEVICE_INDEX: {SELECTED_DEVICE_INDEX}")
    else:
        logging.info("SELECTED_DEVICE_INDEX: None (výchozí zařízení)")
    
    # GROQ API
    logging.info("--- EXTERNÍCH SLUŽBY ---")
    has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
    logging.info(f"GROQ_API_KEY nastaveno: {has_groq_key}")
    
    logging.info("="*60)
    logging.info("INICIALIZACE DOKONČENA")
    logging.info("="*60)

def get_or_create_session_dir(force_new=False):
    """
    Vytvoří novou složku pro seanci nebo vrátí cestu k poslední existující.
    """
    global current_session_dir
    
    if not os.path.exists(SESSIONS_BASE_DIR):
        os.makedirs(SESSIONS_BASE_DIR)

    # UPRAVENO: Pokud chceme novou (Reset tlačítko), vytvoříme ji
    if force_new:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"Session_{timestamp}"
        current_session_dir = os.path.join(SESSIONS_BASE_DIR, session_name)
        os.makedirs(current_session_dir, exist_ok=True)
        setup_logging(current_session_dir) 
        return current_session_dir
    
    # Pokud už máme v paměti cestu, vrátíme ji
    if current_session_dir and os.path.exists(current_session_dir):
        return current_session_dir
    
    # UPRAVENO: Při startu aplikace (nebo reloadu) najdeme poslední existující
    subdirs = glob.glob(os.path.join(SESSIONS_BASE_DIR, 'Session_*'))
    if subdirs:
        # Najdi nejnovější složku
        latest_session = max(subdirs, key=os.path.getctime)
        current_session_dir = latest_session
        setup_logging(current_session_dir) # Aktivujeme logování do této existující složky
        return current_session_dir
    else:
        # Pokud žádná neexistuje (úplně první start), vytvoříme novou
        return get_or_create_session_dir(force_new=True)

# --- NAČTENÍ A PŘÍPRAVA DAT ---
def initialize_data_structures():
    global df_med_items, numeric_items, text_items, df_numeric_state, df_text_state
    
    try:
        df_med_items = pd.read_csv(CSV_DATA_PATH, delimiter=';')
        df_med_items.rename(columns={'DrABCDE': 'Položka'}, inplace=True)       
        
        numeric_items = []
        text_items = []
        
        for _, row in df_med_items.iterrows():
            if pd.notna(row['Referenční hodnoty']):
                ref_val = str(row['Referenční hodnoty'])
                if any(char.isdigit() for char in ref_val):
                    numeric_items.append(row['Položka'])
                else:
                    text_items.append(row['Položka'])
            else:
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

# Inicializace struktur
initialize_data_structures()

# UPRAVENO: Hned při startu inicializujeme session, aby logy někam padaly
get_or_create_session_dir(force_new=False)


# --- LLM PROMPTY ---
numeric_items_str = ', '.join(f'"{item}"' for item in numeric_items)
text_items_str = ', '.join(f'"{item}"' for item in text_items)

system_prompt = f"""
Jsi expert na extrakci lékařských dat z mluveného slova. Tvým úkolem je z textu extrahovat medicínské parametry (vitální funkce), podané léky a nálezy.
Ber v úvahu pouze textový vstup v češtině.

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
"""

# --- INICIALIZACE GROQ KLIENTA ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    logging.warning(f"Nepodařilo se inicializovat GROQ klienta: {e}")
    groq_client = None

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
            
            # Pokud je zapnuto nahrávání, vytvoříme soubor v aktuální session
            if SAVE_CONTINUOUS_AUDIO:
                session_dir = get_or_create_session_dir(force_new=False)
                timestamp = datetime.now().strftime('%H%M%S')
                # UPRAVENO: Název souboru je jasně identifikovatelný v session složce
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
            # UPRAVENO: Používám logging místo print pro konzistenci
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

            with open(tmp_wav_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    model=GROQ_TRANSCRIPTION_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            os.remove(tmp_wav_path)
            # UPRAVENO: Používám logging místo print
            logging.info(f"[GROQ PŘEPIS] {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Chyba při GROQ transkripci: {e}", exc_info=True)
            socketio.emit('processing_error', {'message': f'Chyba při GROQ transkripci: {e}'})
            return None

    def process_audio_chunk(self, data):
        if self.continuous_wave_file:
            self.continuous_wave_file.writeframes(data)

        self.audio_buffer.append(data)
        speech_detected = self.is_speech_detected(data)

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
def get_data_from_ollama(text: str) -> dict | None:
    try:
        response = ollama.chat(
            model=LOCAL_MODEL_NAME,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            options={'temperature': LOCAL_TEMPERATURE, 'response_format': {'type': 'json_object'}}
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        logging.error(f"Chyba při komunikaci s Ollama: {e}")
        return None

def get_data_from_groq(text: str) -> dict | None:
    if not groq_client:
        logging.error("GROQ klient není dostupný.")
        socketio.emit('processing_error', {'message': 'GROQ klient není dostupný.'})
        return None
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"Chyba při komunikaci s GROQ: {e}")
        return None

def save_to_csv():
    """
    UPRAVENO: Ukládá soubory bez časového razítka v názvu, aby se přepisovaly
    a tvořily tak 'aktuální výsledný soubor' v dané session.
    """
    try:
        session_dir = get_or_create_session_dir(force_new=False)
        
        # Pevné názvy pro výsledné soubory
        numeric_csv_path = os.path.join(session_dir, 'vysledny_stav_numeric.csv')
        df_numeric_state.to_csv(numeric_csv_path)
        
        text_csv_path = os.path.join(session_dir, 'vysledny_stav_text.csv')
        df_text_state.to_csv(text_csv_path)
        
        logging.info(f"Tabulky aktualizovány v: {session_dir}")
    except Exception as e:
        logging.error(f"Chyba při ukládání CSV: {e}")

def process_with_llm(text: str):
    global df_numeric_state, df_text_state
    
    get_or_create_session_dir(force_new=False)
    
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
                logging.warning(f"Položka '{item_name}' nenalezena.")
                continue

            if found_item in numeric_items:
                try:
                    if "/" in str(value):
                        df_numeric_state.loc[found_item, timestamp] = str(value)
                    else:
                        df_numeric_state.loc[found_item, timestamp] = float(value)
                except (ValueError, TypeError):
                    logging.warning(f"Nečíselná hodnota '{value}' pro '{found_item}'.")
                    df_numeric_state.loc[found_item, timestamp] = str(value)
            
            elif found_item in text_items:
                df_text_state.loc[found_item, timestamp] = str(value)
        
        emit_table_update(extracted_data)
        save_to_csv() 
        
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
    logging.info("Tabulky odeslány klientovi.")

# --- FLASK ROUTES & SOCKETIO EVENTS ---
@app.route('/')
def index():
    html_numeric = df_numeric_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    html_text = df_text_state.fillna('').to_html(classes="table table-striped table-hover", border=0)
    return render_template('index.html', 
                           numeric_table=html_numeric, 
                           text_table=html_text, 
                           provider=AI_PROVIDER,
                           save_audio=SAVE_CONTINUOUS_AUDIO)

@app.route('/process', methods=['POST'])
def process_text():
    user_text = request.form['text_input']
    if user_text:
        # UPRAVENO: Použití logging.info pro konzistenci
        logging.info(f"[MANUÁLNÍ VSTUP] {user_text}")
        threading.Thread(target=process_with_llm, args=(user_text,)).start()
    return redirect(url_for('index'))

@socketio.on('set_ai_provider')
def set_ai_provider(data):
    global AI_PROVIDER
    new_provider = data.get('provider')
    if new_provider in ['local', 'groq']:
        AI_PROVIDER = new_provider
        logging.info(f"AI provider změněn na: {AI_PROVIDER}")
        emit('provider_changed', {'provider': AI_PROVIDER})

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
    logging.info("Požadavek na reset session.")
    # Zde force_new=True vytvoří NOVOU složku
    new_dir = get_or_create_session_dir(force_new=True)
    
    initialize_data_structures()
    
    emit_table_update(extracted_data={})
    logging.info(f"Session resetována. Nová složka: {os.path.basename(new_dir)}")
    emit('session_reset_complete', {'message': f'Nová seance zahájena: {os.path.basename(new_dir)}'})


@socketio.on('start_recording')
def handle_start_recording():
    global speech_processor, is_recording
    with recording_lock:
        if not is_recording:
            # Použijeme existující session, nevytváříme novou
            get_or_create_session_dir(force_new=False)
            
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
        logging.info("Ukončuji aplikaci, zastavuji nahrávání.")
        is_recording = False

atexit.register(cleanup_app)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        logging.error(f"CHYBA: Vosk model nenalezen: {MODEL_PATH}")
    if not os.environ.get("GROQ_API_KEY"):
        logging.warning("Chybí GROQ_API_KEY.")

    socketio.run(app, debug=False, host='0.0.0.0', port=5050)