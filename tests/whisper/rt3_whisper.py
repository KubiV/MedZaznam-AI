import whisper
import pyaudio
import numpy as np
import wave
import tempfile
import time
import os
from collections import deque
import threading
from datetime import datetime

# 🔧 Parametry zvuku
RATE = 16000
CHUNK = 1024
CHANNELS = 1

# 🔊 VAD parametry
SILENCE_THRESHOLD = 200      # Práh pro detekci ticha (experimentujte s hodnotami 200-2000)
MIN_SPEECH_DURATION = 0.5    # Minimální délka řeči v sekundách pro zahájení nahrávání
SILENCE_DURATION = 1.5       # Délka ticha v sekundách pro ukončení věty
MAX_RECORDING_TIME = 30      # Maximální délka nahrávání v sekundách (ochrana proti nekonečnému nahrávání)

# 📁 Složka pro uložení nahrávek a přepisů
OUTPUT_DIR = "recordings"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 🤖 Whisper model - pro češtinu doporučuji minimálně "base"
model = whisper.load_model("base")

def list_audio_devices():
    """Vypíše seznam dostupných audio zařízení"""
    p = pyaudio.PyAudio()
    print("\n🎙️  Dostupná audio zařízení:")
    print("=" * 60)

    default_input = None
    try:
        default_input = p.get_default_input_device_info()['index']
    except:
        pass

    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Pouze vstupní zařízení
                marker = " [VÝCHOZÍ]" if i == default_input else ""
                print(f"📱 ID {i}: {info['name']}{marker}")
                print(f"   Kanály: {info['maxInputChannels']}, Frekvence: {info['defaultSampleRate']} Hz")
                print("-" * 40)
        except Exception as e:
            continue

    p.terminate()
    print(f"🎯 Program použije zařízení ID: {default_input}")
    return default_input

class VoiceActivityDetector:
    def __init__(self, device_id=None):
        self.device_id = device_id
        self.is_speaking = False
        self.speech_frames = []
        self.silence_counter = 0
        self.speech_counter = 0
        self.recording_count = 0

        # Kontinuální záznam pro backup
        self.continuous_frames = []
        self.continuous_recording = True
        self.backup_thread = None

        # Audio buffer pro analýzu
        self.audio_buffer = deque(maxlen=int(RATE * MAX_RECORDING_TIME / CHUNK))

        # Soubor pro přepisy s časovými známkami
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript_file = os.path.join(OUTPUT_DIR, f"transcript_{timestamp}.txt")

        # Inicializace souboru s přepisy
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"🎙️ TRANSKRIPCE NAHRÁVKY - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

        # PyAudio inicializace
        self.p = pyaudio.PyAudio()

        # Zobrazení informací o použitém zařízení
        if device_id is not None:
            try:
                device_info = self.p.get_device_info_by_index(device_id)
                print(f"🎤 Používám zařízení: {device_info['name']}")
            except:
                print(f"⚠️  Zařízení ID {device_id} není dostupné, používám výchozí")
                device_id = None

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=CHUNK
        )

        # Spuštění kontinuálního záznamu
        self.start_continuous_recording()

        print("🎙️  VAD aktivní - čekám na řeč... (CTRL+C pro ukončení)")
        print(f"📊 Práh ticha: {SILENCE_THRESHOLD}, Min. řeč: {MIN_SPEECH_DURATION}s, Max. tichá pauza: {SILENCE_DURATION}s")
        print(f"💾 Přepisy se ukládají do: {self.transcript_file}")
        print(f"📁 Nahrávky se ukládají do: {OUTPUT_DIR}/")

    def start_continuous_recording(self):
        """Spustí kontinuální záznam pro backup"""
        self.backup_thread = threading.Thread(target=self.continuous_backup_worker, daemon=True)
        self.backup_thread.start()

    def continuous_backup_worker(self):
        """Worker pro ukládání kontinuálního záznamu"""
        backup_duration = 60  # Ukládej každou minutu
        backup_counter = 0

        while self.continuous_recording:
            time.sleep(backup_duration)
            if len(self.continuous_frames) > 0:
                backup_counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = os.path.join(OUTPUT_DIR, f"backup_part_{backup_counter:03d}_{timestamp}.wav")

                try:
                    # Uložení backup souboru
                    with wave.open(backup_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(self.continuous_frames))

                    print(f"💾 Backup uložen: {backup_filename}")
                    self.continuous_frames = []  # Vyčisti buffer
                except Exception as e:
                    print(f"❌ Chyba při ukládání backup: {e}")

    def calculate_rms(self, audio_data):
        """Výpočet RMS (Root Mean Square) pro detekci hlasitosti"""
        try:
            if not audio_data or len(audio_data) == 0:
                return 0.0

            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Kontrola prázdného pole
            if len(audio_array) == 0:
                return 0.0

            # Převod na float pro bezpečné výpočty
            audio_float = audio_array.astype(np.float64)

            # Výpočet RMS s ochranou proti nevalidním hodnotám
            mean_square = np.mean(audio_float**2)

            # Ochrana proti záporným nebo NaN hodnotám
            if mean_square <= 0 or np.isnan(mean_square) or np.isinf(mean_square):
                return 0.0

            rms = np.sqrt(mean_square)

            # Finální kontrola validity
            if np.isnan(rms) or np.isinf(rms):
                return 0.0

            return float(rms)

        except Exception as e:
            print(f"⚠️  Chyba při výpočtu RMS: {e}")
            return 0.0

    def is_speech_detected(self, audio_data):
        """Detekce řeči na základě RMS hodnoty"""
        rms = self.calculate_rms(audio_data)

        # Debug info - můžete zakomentovat po vyladění
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 1

        # Výpis RMS hodnot každých 50 chunků pro ladění prahu
        if self.debug_counter % 50 == 0:
            print(f"🔊 Aktuální RMS: {rms:.1f} (práh: {SILENCE_THRESHOLD})")

        return rms > SILENCE_THRESHOLD

    def save_audio_to_wav(self, frames, prefix="speech"):
        """Uložení audio dat do WAV souboru s časovou známkou"""
        self.recording_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"{prefix}_{self.recording_count:03d}_{timestamp}.wav")

        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"💾 Nahrávka uložena: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Chyba při ukládání: {e}")
            return None

    def save_transcript(self, text, timestamp, audio_filename):
        """Uložení přepisu do souboru s časovou známkou"""
        try:
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] - {audio_filename}\n")
                f.write(f"📝 {text}\n")
                f.write("-" * 60 + "\n\n")
        except Exception as e:
            print(f"❌ Chyba při ukládání přepisu: {e}")

    def transcribe_audio(self, wav_filename):
        """Transkripce audio souboru pomocí Whisper"""
        try:
            print("🧠 Zpracovávám řeč...")

            # Whisper transkripce s word-level timestamps
            result = model.transcribe(
                wav_filename,
                language="cs",
                task="transcribe",
                temperature=0.0,
                best_of=1,
                beam_size=1,
                word_timestamps=True,  # Povolí časové známky na úrovni slov
                verbose=False
            )

            #result = model.transcribe(
            #    wav_filename,
            #    language="cs",
            #    fp16=False,
            #    beam_size=1,     # Rychlejší
            #    best_of=1,       # Rychlejší
            #    temperature=0    # Deterministický
            #)

            # Vyčištění textu
            text = result["text"].strip()
            current_time = datetime.now().strftime("%H:%M:%S")

            if text and len(text) > 1:
                print(f"👂 [{current_time}] {text}")
                print("-" * 50)

                # Uložení do souboru
                self.save_transcript(text, current_time, os.path.basename(wav_filename))

                # Detailní výpis s časovými známkami slov (pokud jsou dostupné)
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment:
                            word_details = []
                            for word_info in segment["words"]:
                                word = word_info.get("word", "").strip()
                                start_time = word_info.get("start", 0)
                                if word:
                                    word_details.append(f"{word}[{start_time:.1f}s]")

                            if word_details:
                                detailed_text = " ".join(word_details)
                                print(f"🕐 Detailní timing: {detailed_text}")

                                # Uložení detailního timingu
                                with open(self.transcript_file, 'a', encoding='utf-8') as f:
                                    f.write(f"🕐 Timing: {detailed_text}\n\n")
            else:
                print("🤐 Žádná srozumitelná řeč nebyla detekována")

        except Exception as e:
            print(f"❌ Chyba při transkripci: {e}")

    def process_audio_chunk(self, data):
        """Zpracování jednoho audio chunku"""
        # Přidání do kontinuálního záznamu
        self.continuous_frames.append(data)

        speech_detected = self.is_speech_detected(data)

        # Přidání do bufferu pro případné uložení
        self.audio_buffer.append(data)

        if speech_detected:
            # Detekována řeč
            self.silence_counter = 0
            self.speech_counter += 1

            if not self.is_speaking:
                # Začátek řeči - kontrola minimální délky
                required_chunks = int(MIN_SPEECH_DURATION * RATE / CHUNK)
                if self.speech_counter >= required_chunks:
                    self.is_speaking = True
                    print("🗣️  Začátek řeči detekován - nahrávám...")
                    # Začni ukládat audio od začátku detekované řeči
                    buffer_start = max(0, len(self.audio_buffer) - self.speech_counter)
                    self.speech_frames = list(self.audio_buffer)[buffer_start:]

            if self.is_speaking:
                # Pokračování v nahrávání
                self.speech_frames.append(data)

                # Ochrana proti příliš dlouhému nahrávání
                if len(self.speech_frames) > int(MAX_RECORDING_TIME * RATE / CHUNK):
                    print("⏰ Dosažena maximální délka nahrávání - ukončujem")
                    self.finalize_speech()

        else:
            # Detekováno ticho
            self.speech_counter = max(0, self.speech_counter - 1)  # Postupné snižování

            if self.is_speaking:
                self.silence_counter += 1
                self.speech_frames.append(data)  # Zahrň i tiché části pro kontext

                # Kontrola konce věty
                required_silence_chunks = int(SILENCE_DURATION * RATE / CHUNK)
                if self.silence_counter >= required_silence_chunks:
                    print("🔚 Konec věty detekován")
                    self.finalize_speech()

    def finalize_speech(self):
        """Dokončení nahrávání a spuštění transkripce"""
        if len(self.speech_frames) > 0:
            # Uložení do WAV souboru
            wav_file = self.save_audio_to_wav(self.speech_frames, "speech")

            if wav_file:
                # Spuštění transkripce v separátním vlákně (neblokující)
                transcription_thread = threading.Thread(
                    target=self.transcribe_audio,
                    args=(wav_file,),
                    daemon=True
                )
                transcription_thread.start()

        # Reset stavu
        self.is_speaking = False
        self.speech_frames = []
        self.silence_counter = 0
        self.speech_counter = 0
        print("🎯 Čekám na další řeč...")

    def run(self):
        """Hlavní smyčka pro kontinuální poslouchání"""
        try:
            print("🔧 Kalibrace mikrofonu - změřím úroveň šumu...")

            # Krátká kalibrace pro zjištění úrovně pozadového šumu
            noise_samples = []
            for i in range(20):  # 20 vzorků pro stanovení pozadí
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    rms = self.calculate_rms(data)
                    if rms > 0:
                        noise_samples.append(rms)
                except:
                    continue

            if noise_samples:
                avg_noise = np.mean(noise_samples)
                suggested_threshold = max(avg_noise * 3, 300)  # 3x hlasitější než šum, min. 300
                print(f"📈 Průměrný šum: {avg_noise:.1f}")
                print(f"💡 Doporučený práh: {suggested_threshold:.1f} (aktuální: {SILENCE_THRESHOLD})")
                print("🎯 Pro změnu prahu upravte SILENCE_THRESHOLD v kódu")

            print("✅ Kalibrace dokončena - začínám poslouchat...")

            while True:
                # Čtení audio dat z mikrofonu
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)

                    # Kontrola validity dat před zpracováním
                    if data and len(data) == CHUNK * 2:  # 2 bytes per sample pro 16-bit
                        self.process_audio_chunk(data)
                    else:
                        print("⚠️  Nevalidní audio data - přeskakuji chunk")
                        continue

                except Exception as e:
                    print(f"⚠️  Chyba při čtení audio: {e}")
                    time.sleep(0.1)
                    continue

        except KeyboardInterrupt:
            print("\n🛑 Ukončeno uživatelem.")

            # Pokud byla řeč v průběhu, dokončí ji
            if self.is_speaking and len(self.speech_frames) > 0:
                print("📝 Dokončuji posledni nahrávání...")
                self.finalize_speech()
                time.sleep(3)  # Počkej na dokončení transkripce

        finally:
            self.cleanup()

    def cleanup(self):
        """Vyčištění zdrojů"""
        print("🧹 Uklízím zdroje...")

        # Zastavení kontinuálního záznamu
        self.continuous_recording = False

        # Uložení posledního backup souboru
        if len(self.continuous_frames) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_backup = os.path.join(OUTPUT_DIR, f"final_backup_{timestamp}.wav")
            try:
                with wave.open(final_backup, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.continuous_frames))
                print(f"💾 Finální backup uložen: {final_backup}")
            except Exception as e:
                print(f"❌ Chyba při ukládání finálního backup: {e}")

        # Dokončení přepisu
        try:
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(f"\n🏁 KONEC TRANSKRIPCE - {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            print(f"📄 Přepis dokončen a uložen: {self.transcript_file}")
        except:
            pass

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# 🚀 Spuštění programu
if __name__ == "__main__":
    # Zobrazení dostupných audio zařízení
    default_device = list_audio_devices()

    print("\n" + "="*60)
    user_input = input("💬 Chcete použít jiné zařízení? Zadejte ID nebo stiskněte Enter pro výchozí: ")

    device_id = None
    if user_input.strip().isdigit():
        device_id = int(user_input.strip())
        print(f"✅ Budu používat zařízení ID: {device_id}")
    else:
        print(f"✅ Budu používat výchozí zařízení")

    print("="*60)

    vad = VoiceActivityDetector(device_id=device_id)
    vad.run()