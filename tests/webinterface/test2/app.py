import ollama
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

# --- LOGGING SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CSV_DIR = os.path.join(BASE_DIR, 'csv_logs')

try:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    print(f"Logs directory: {LOGS_DIR}")
    print(f"CSV directory: {CSV_DIR}")
except Exception as e:
    print(f"Error creating directories: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'voice_inventory.log')),
        logging.StreamHandler()
    ]
)

# --- AUDIO PARAMETERS ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 150  # Lowered for better speech detection sensitivity
MIN_SPEECH_DURATION = 0.5
SILENCE_DURATION = 1.5
MAX_RECORDING_TIME = 15

# --- VOSK MODEL ---
MODEL_PATH = "/Users/jakubvavra/Documents/GitHub/Automonitoring-with-AI/tests/vosk/vosk-model-small-cs-0.4-rhasspy"

# --- FLASK AND SOCKETIO INITIALIZATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- DATA ---
PREDEFINED_ITEMS = [
    'Mléko', 'Chléb', 'Máslo', 'Sýr', 'Vejce', 'Jogurt', 'Ovoce', 'Zelenina',
    'Brambory', 'Těstoviny', 'Rýže', 'Maso', 'Uzeniny', 'Káva', 'Čaj', 'Cukr',
    'Mouka', 'Olej', 'Toaletní papír', 'Mýdlo', 'Prací prostředek', 'Rohlík', 'Houska'
]

PREDEFINED_ITEMS_1 = [
    'SpO2', 'Srdeční frekvence', 'Krevní tlak', 'Dechová frekvence', 'Teplota',
    'Bolest NRS', 'Dušnost', 'Kyslíková terapie', 'Léky', 'Infuzní terapie',
    'Anamnéza', 'Fyzikální vyšetření', 'Příznaky'
]

df_state = pd.DataFrame(
    {'Počáteční stav': 0},
    index=PREDEFINED_ITEMS
)
df_state.index.name = "Položka"

# --- GLOBAL VARIABLES & LOCKS FOR THREAD SAFETY ---
speech_processor = None
is_recording = False
recording_lock = threading.Lock()  # ADDED: Lock for thread-safe access to globals

# --- LLM SYSTEM PROMPT ---
system_prompt = """
Jsi expert na extrakci dat pro systém sledování zásob. Z textu extrahuj potraviny, jejich počet a typ operace.

Pravidla:
1. Výstup musí být VŽDY a POUZE platný JSON objekt.
2. JSON obsahuje klíč 'operace', který může mít dvě hodnoty:
   - 'prirustek': Pokud text popisuje přidání nebo odebrání položek (např. "koupil jsem", "přidal jsem", "vrátil jsem"). Zde použij záporná čísla pro odebrání.
   - 'nastaveni': Pokud text popisuje finální, absolutní stav (např. "mám celkem", "zůstalo mi", "aktuální stav je").
3. Druhý klíč je 'polozky', což je slovník, kde klíče jsou názvy potravin v 1. pádu jednotného čísla a hodnoty jsou čísla.
4. Pokud text neobsahuje informace o potravinách nebo operaci, vrať prázdný slovník 'polozky' a operaci 'none'.

Příklady:
Uživatel: "Koupil jsem 2 mléka a 3 rohlíky"
Výstup: {"operace": "prirustek", "polozky": {"mléko": 2, "rohlík": 3}}

Uživatel: "Mám teď celkem 5 vajec"
Výstup: {"operace": "nastaveni", "polozky": {"vejce": 5}}

Uživatel: "Halo"
Výstup: {"operace": "none", "polozky": {}}
"""

system_prompt_1 = """

Jsi expert na extrakci lékařských dat.
Tvým úkolem je z textu extrahovat medicínské parametry, nálezy a stavy pacienta podle standardizovaných položek (DrABCDE, vitální funkce, anamnéza, fyzikální vyšetření, léčba, příznaky).

Pravidla:
	1.	Výstup musí být VŽDY a POUZE platný JSON objekt.
	2.	JSON obsahuje dva klíče:
	•	"operace":
	•	"prirustek" – pokud text popisuje přidání, provedení nebo změnu stavu (např. „nasadil jsem kyslík“, „zhoršila se dušnost“, „přidal jsem léky“).
	•	"nastaveni" – pokud text popisuje aktuální, finální nebo absolutní stav (např. „pacient má SpO2 98 %“, „tepová frekvence je 120/min“, „bolest 5/10“).
	•	"none" – pokud text neobsahuje žádnou relevantní informaci o zdravotním stavu nebo intervenci.
	•	"polozky": slovník, kde klíče jsou názvy lékařských položek v 1. pádě jednotného čísla (např. „SpO2“, „srdeční frekvence“, „bolest NRS“, „krevní tlak“, „dušnost“) a hodnoty jsou číselné nebo textové údaje podle toho, co je ve vstupu.
	3.	Pokud text obsahuje více různých údajů, ulož je všechny do "polozky".
	4.	Pokud není jasná hodnota (např. jen „pacient má bolesti“), ulož ji jako řetězec. Pokud je uvedena číselně (např. „bolest 6/10“), ulož číslo.
	5.	Názvy položek ber vždy z předem definovaného seznamu (DrABCDE, anamnéza, vyšetření, příznaky, léčba atd. – viz přiložená CSV). Pokud se objeví synonymum, normalizuj ho (např. „tep“ → „srdeční frekvence“, „saturace“ → „SpO2“).

Příklady:

Uživatel: „Pacient má SpO2 92 % a tep 120/min“
Výstup: {"operace": "nastaveni", "polozky": {"SpO2": 92, "srdeční frekvence": 120}}

Uživatel: „Nasadil jsem kyslíkovou terapii a přidal infuzi“
Výstup: {"operace": "prirustek", "polozky": {"oxygenoterapie": "zahájena", "léky": "infuzní terapie"}}

Uživatel: „Pacient si stěžuje na bolesti břicha“
Výstup: {"operace": "nastaveni", "polozky": {"bolest": "břicho"}}

Uživatel: „Halo“
Výstup: {"operace": "none", "polozky": {}}

"""

class VoiceSpeechProcessor:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.p = None
        self.stream = None
        self.is_speaking = False
        self.speech_frames = []
        self.silence_counter = 0
        self.speech_counter = 0
        self.audio_buffer = deque(maxlen=int(RATE * MAX_RECORDING_TIME / CHUNK))
        self.initialize_vosk()

    def initialize_vosk(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Vosk model nenalezen: {MODEL_PATH}")
            self.model = Model(MODEL_PATH)
            self.recognizer = KaldiRecognizer(self.model, RATE)
            self.recognizer.SetWords(True)
            logging.info("Vosk model úspěšně načten")
            print("[INFO]: Vosk model úspěšně načten")
        except Exception as e:
            logging.error(f"Chyba při inicializaci Vosk: {e}")
            print(f"[CHYBA VOSK]: {e}")
            raise

    def initialize_audio(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=None
            )
            logging.info("Audio stream inicializován")
            print("[INFO]: Audio stream inicializován")
        except Exception as e:
            logging.error(f"Chyba při inicializaci audio: {e}")
            print(f"[CHYBA AUDIO]: {e}")
            self.cleanup()
            raise

    def calculate_rms(self, audio_data):
        try:
            if not audio_data or len(audio_data) == 0:
                return 0.0
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            audio_float = audio_array.astype(np.float64)
            mean_square = np.mean(audio_float**2)
            if mean_square <= 0 or np.isnan(mean_square) or np.isinf(mean_square):
                return 0.0
            rms = np.sqrt(mean_square)
            if np.isnan(rms) or np.isinf(rms):
                return 0.0
            return float(rms)
        except Exception as e:
            logging.error(f"Chyba při výpočtu RMS: {e}")
            print(f"[CHYBA RMS]: {e}")
            return 0.0

    def is_speech_detected(self, audio_data):
        rms = self.calculate_rms(audio_data)
        logging.debug(f"RMS: {rms}")
        return rms > SILENCE_THRESHOLD

    def transcribe_audio(self, audio_frames):
        try:
            if not self.recognizer:
                logging.error("Recognizer není inicializován")
                print("[CHYBA PŘEPISU]: Recognizer není inicializován")
                return None
            if not audio_frames:
                logging.warning("Žádné audio rámce k přepisu")
                print("[PŘEPIS]: Žádné audio rámce k přepisu")
                return None

            self.recognizer.Reset()
            audio_data = b''.join(audio_frames)
            chunk_size = 4000
            results = []

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if self.recognizer.AcceptWaveform(chunk):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text', '').strip():
                        results.append(result['text'].strip())

            final_result = json.loads(self.recognizer.FinalResult())
            if final_result.get('text', '').strip():
                results.append(final_result['text'].strip())

            full_text = ' '.join(results).strip()
            if full_text and len(full_text) > 1:
                logging.info(f"Přepsáno: {full_text}")
                print(f"[PŘEPIS]: {full_text}")
                return full_text
            else:
                logging.warning("Žádný přepis nenalezen")
                print("[PŘEPIS]: Žádný text nerozpoznán")
                return None
        except Exception as e:
            logging.error(f"Chyba při transkripci: {e}")
            print(f"[CHYBA PŘEPISU]: {e}")
            return None

    def process_audio_chunk(self, data):
        try:
            if not self.recognizer:
                logging.error("Recognizer není inicializován při zpracování chunku")
                print("[CHYBA AUDIO]: Recognizer není inicializován")
                return

            speech_detected = self.is_speech_detected(data)
            self.audio_buffer.append(data)

            if speech_detected:
                self.silence_counter = 0
                self.speech_counter += 1

                if not self.is_speaking:
                    required_chunks = int(MIN_SPEECH_DURATION * RATE / CHUNK)
                    if self.speech_counter >= required_chunks:
                        self.is_speaking = True
                        socketio.emit('speech_start')
                        logging.info("Začátek řeči detekován")
                        print("[INFO]: Začátek řeči detekován")
                        buffer_start = max(0, len(self.audio_buffer) - self.speech_counter)
                        self.speech_frames = list(self.audio_buffer)[buffer_start:]

                if self.is_speaking:
                    self.speech_frames.append(data)

                    if len(self.speech_frames) > int(MAX_RECORDING_TIME * RATE / CHUNK):
                        logging.info("Dosažena max délka - ukončuji")
                        print("[INFO]: Maximální délka nahrávky dosažena")
                        self.finalize_speech()

            else:
                self.speech_counter = max(0, self.speech_counter - 1)

                if self.is_speaking:
                    self.silence_counter += 1
                    self.speech_frames.append(data)

                    required_silence_chunks = int(SILENCE_DURATION * RATE / CHUNK)
                    if self.silence_counter >= required_silence_chunks:
                        logging.info("Konec věty detekován")
                        print("[INFO]: Konec věty detekován")
                        self.finalize_speech()
        except Exception as e:
            logging.error(f"Chyba při zpracování audio chunku: {e}")
            print(f"[CHYBA AUDIO]: {e}")

    def finalize_speech(self):
        try:
            if len(self.speech_frames) > 0 and self.recognizer:
                frames_to_process = self.speech_frames.copy()

                def process_transcription_and_llm():
                    try:
                        transcribed_text = self.transcribe_audio(frames_to_process)
                        if transcribed_text:
                            socketio.emit('transcription_result', {'text': transcribed_text})
                            print(f"[ODESLÁN PŘEPIS]: {transcribed_text}")
                            process_with_llm(transcribed_text)
                        else:
                            socketio.emit('transcription_result', {'text': 'Žádný text nerozpoznán'})
                            socketio.emit('processing_error', {'message': 'Nepodařilo se rozpoznat řeč.'})
                            print("[ODESLÁN PŘEPIS]: Žádný text nerozpoznán")
                    except Exception as e:
                        logging.error(f"NECHYCENÁ VÝJIMKA ve vláknu zpracování: {e}", exc_info=True)
                        print(f"[CHYBA VLÁKNA]: {e}")
                        socketio.emit('processing_error', {'message': f'Vnitřní chyba serveru při zpracování: {e}'})

                threading.Thread(target=process_transcription_and_llm, daemon=True).start()
            else:
                logging.warning("Žádné rámce nebo recognizer není k dispozici pro přepis")
                socketio.emit('transcription_result', {'text': 'Žádný text nerozpoznán'})
                socketio.emit('processing_error', {'message': 'Nebyly zaznamenány žádné zvukové rámce.'})
                print("[ODESLÁN PŘEPIS]: Žádný text nerozpoznán")

            self.is_speaking = False
            self.speech_frames = []
            self.silence_counter = 0
            self.speech_counter = 0
            socketio.emit('speech_end')
        except Exception as e:
            logging.error(f"Chyba při finalizaci řeči: {e}")
            print(f"[CHYBA FINALIZACE]: {e}")

    def start_listening(self):
        global is_recording
        try:
            self.initialize_audio()
            logging.info("Zahajuji nahrávání")
            print("[INFO]: Zahajuji nahrávání")

            while is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    if data and is_recording:
                        self.process_audio_chunk(data)
                except IOError as e:
                    logging.warning(f"Chyba I/O při čtení audio (očekáváno při zastavení): {e}")
                    break
                except Exception as e:
                    logging.error(f"Chyba při čtení audio: {e}")
                    print(f"[CHYBA ČTENÍ AUDIO]: {e}")
                    continue
        except Exception as e:
            logging.error(f"Kritická chyba při poslouchání: {e}", exc_info=True)
            print(f"[CHYBA POSLOUCHÁNÍ]: {e}")
        finally:
            self.cleanup()
            with recording_lock:
                is_recording = False

    def stop_listening(self):
        global is_recording
        with recording_lock:
             is_recording = False

        if self.is_speaking and len(self.speech_frames) > 0 and self.recognizer:
            self.finalize_speech()

    def cleanup(self):
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
            if self.stream:
                self.stream.close()
            if self.p:
                self.p.terminate()

            self.stream = None
            self.p = None
            logging.info("Audio stream ukončen")
            print("[INFO]: Audio stream ukončen")
        except Exception as e:
            logging.error(f"Chyba při čištění audio: {e}")
            print(f"[CHYBA ČIŠTĚNÍ]: {e}")

def save_to_csv():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(CSV_DIR, f'inventory_{timestamp}.csv')
        df_state.to_csv(csv_path)
        logging.info(f"Tabulka uložena do CSV: {csv_path}")
        print(f"[INFO]: Tabulka uložena do CSV: {csv_path}")
    except Exception as e:
        logging.error(f"Chyba při ukládání CSV: {e}")
        print(f"[CHYBA CSV]: {e}")

def get_data_from_ollama(text: str) -> dict | None:
    logging.info(f"Zpracovávám LLM: {text}")
    print(f"[LLM VSTUP]: {text}")
    try:
        response = ollama.chat(
            model='gemma3:1b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            options={'response_format': {'type': 'json_object'}}
        )

        json_text = response['message']['content']
        logging.info(f"LLM surový výstup: {json_text}")
        print(f"[LLM SUROVÝ VÝSTUP]: {json_text}")
        start = json_text.find('{')
        end = json_text.rfind('}') + 1

        if start != -1 and end != -1:
            result = json.loads(json_text[start:end])
            print(f"[LLM VÝSTUP]: {result}")
            return result
        else:
            logging.warning("LLM nevrátil platný JSON")
            print("[CHYBA LLM]: Neplatný JSON formát")
            return None
    except Exception as e:
        logging.error(f"Chyba při komunikaci s Ollama: {e}")
        print(f"[CHYBA OLLAMA]: {e}")
        return None

def process_with_llm(text: str):
    global df_state

    extracted_data = get_data_from_ollama(text)

    if extracted_data and 'operace' in extracted_data and 'polozky' in extracted_data:
        if extracted_data['operace'] != 'none' and extracted_data['polozky']:
            logging.info(f"LLM extrahovala data: {extracted_data}")
            print(f"[LLM DATA]: {extracted_data}")

            timestamp = datetime.now().strftime('%H:%M:%S')
            last_column = df_state.columns[-1]
            df_state[timestamp] = df_state[last_column].copy()

            operation = extracted_data['operace']
            items = extracted_data['polozky']

            for item_name, value in items.items():
                item_name_clean = item_name.strip().capitalize()

                if item_name_clean not in df_state.index:
                    new_row = pd.Series(0, index=df_state.columns, name=item_name_clean)
                    df_state.loc[item_name_clean] = new_row

                try:
                    numeric_value = int(value)
                    if operation == 'prirustek':
                        df_state.loc[item_name_clean, timestamp] = df_state.loc[item_name_clean, last_column] + numeric_value
                    elif operation == 'nastaveni':
                        df_state.loc[item_name_clean, timestamp] = numeric_value
                except (ValueError, TypeError):
                    logging.warning(f"Neplatná hodnota '{value}' pro položku '{item_name_clean}'. Přeskakuji.")
                    continue

            save_to_csv()
            table_html = df_state.to_html(classes="table table-striped table-hover", border=0)
            socketio.emit('table_update', {'table': table_html, 'threat': None, 'extracted_data': extracted_data})
            print("[INFO]: Tabulka aktualizována")
        else:
            logging.info("Žádné položky k aktualizaci (operace: none nebo prázdné položky)")
            socketio.emit('processing_error', {'message': 'Nerozuměl jsem, žádné položky k aktualizaci.'})
            print("[INFO]: Žádné položky k aktualizaci")
    else:
        logging.warning("LLM neextrahovala platná data")
        socketio.emit('processing_error', {'message': 'Nepodařilo se zpracovat text pomocí AI.'})
        print("[CHYBA]: LLM neextrahovala platná data")

# --- FLASK ROUTES ---
@app.route('/')
def index():
    table_html = df_state.to_html(classes="table table-striped table-hover", border=0)
    return render_template('voice_index_2.html', table=table_html)

@app.route('/process', methods=['POST'])
def process_text():
    user_text = request.form['text_input']
    if user_text:
        threading.Thread(target=process_with_llm, args=(user_text,)).start()
    return redirect(url_for('index'))

@app.route('/test_socket')
def test_socket():
    socketio.emit('test_event', {'message': 'SocketIO test event'})
    return "Test SocketIO event emitted. Check browser console."

# --- SOCKETIO EVENTS (FIXED FOR STABILITY) ---
@socketio.on('start_recording')
def handle_start_recording():
    global speech_processor, is_recording
    with recording_lock:
        if not is_recording:
            is_recording = True
            logging.info("Spouštím nahrávání")
            print("[INFO]: Spouštím nahrávání")
            speech_processor = VoiceSpeechProcessor()

            def start_listening_thread():
                try:
                    speech_processor.start_listening()
                except Exception as e:
                    logging.critical(f"NECHYCENÁ VÝJIMKA ve vláknu nahrávání: {e}", exc_info=True)
                    global is_recording
                    with recording_lock:
                        is_recording = False

            threading.Thread(target=start_listening_thread, daemon=True).start()
            emit('recording_started')
        else:
            logging.warning("Pokus o spuštění nahrávání, které již běží.")
            emit('recording_already_active')

@socketio.on('stop_recording')
def handle_stop_recording():
    global speech_processor, is_recording
    with recording_lock:
        if is_recording and speech_processor:
            logging.info("Zastavuji nahrávání")
            print("[INFO]: Zastavuji nahrávání")
            speech_processor.stop_listening()
            emit('recording_stopped')
        else:
            logging.warning("Pokus o zastavení nahrávání, které není aktivní.")
            emit('recording_not_active')

# Cleanup on exit
def cleanup_app():
    global speech_processor, is_recording
    logging.info("Zahajuji čištění aplikace při ukončení...")
    if is_recording and speech_processor:
        speech_processor.stop_listening()
    logging.info("Čištění aplikace dokončeno.")
    print("[INFO]: Application cleanup completed")

atexit.register(cleanup_app)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"❌ CHYBA: Vosk model nenalezen na cestě: {MODEL_PATH}")
        print("💡 Stáhněte model z: https://alphacephei.com/vosk/models")
        exit(1)

    logging.info("Spouštím Voice Inventory aplikaci")
    print("[INFO]: Spouštím Voice Inventory aplikaci")
    socketio.run(app, debug=True, host='0.0.0.0', port=5050, allow_unsafe_werkzeug=True)