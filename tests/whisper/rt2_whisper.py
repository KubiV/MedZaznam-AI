import whisper
import pyaudio
import numpy as np
import wave
import tempfile
import time
from collections import deque
import threading

# 🔧 Parametry zvuku
RATE = 16000
CHUNK = 1024
CHANNELS = 1

# 🔊 VAD parametry
SILENCE_THRESHOLD = 500      # Práh pro detekci ticha (experimentujte s hodnotami 200-2000)
MIN_SPEECH_DURATION = 0.5    # Minimální délka řeči v sekundách pro zahájení nahrávání
SILENCE_DURATION = 1.5       # Délka ticha v sekundách pro ukončení věty
MAX_RECORDING_TIME = 30      # Maximální délka nahrávání v sekundách (ochrana proti nekonečnému nahrávání)

# 🤖 Whisper model - pro češtinu doporučuji minimálně "base"
model = whisper.load_model("medium")  # Můžete zvolit i "tiny", "base", "small", "large" podle potřeby

class VoiceActivityDetector:
    def __init__(self):
        self.is_speaking = False
        self.speech_frames = []
        self.silence_counter = 0
        self.speech_counter = 0

        # Audio buffer pro analýzu
        self.audio_buffer = deque(maxlen=int(RATE * MAX_RECORDING_TIME / CHUNK))

        # PyAudio inicializace
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("🎙️  VAD aktivní - čekám na řeč... (CTRL+C pro ukončení)")
        print(f"📊 Práh ticha: {SILENCE_THRESHOLD}, Min. řeč: {MIN_SPEECH_DURATION}s, Max. tichá pauza: {SILENCE_DURATION}s")

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

    def save_audio_to_wav(self, frames):
        """Uložení audio dat do dočasného WAV souboru"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            wf = wave.open(tmpfile.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return tmpfile.name

    def transcribe_audio(self, wav_filename):
        """Transkripce audio souboru pomocí Whisper"""
        try:
            print("🧠 Zpracovávám řeč...")
            result = model.transcribe(
                wav_filename,
                language="cs",
                task="transcribe",
                # Dodatečné parametry pro lepší výsledky
                temperature=0.0,  # Deterministický výstup
                best_of=1,       # Rychlejší zpracování
                beam_size=1,     # Rychlejší zpracování
                word_timestamps=False
            )

            # Vyčištění textu (odstranění prázdných výsledků)
            text = result["text"].strip()
            if text and len(text) > 2:  # Ignoruj velmi krátké nebo prázdné výsledky
                print(f"👂 Přepis: {text}")
                print("-" * 50)
            else:
                print("🤐 Žádná srozumitelná řeč nebyla detekována")

        except Exception as e:
            print(f"❌ Chyba při transkripci: {e}")

    def process_audio_chunk(self, data):
        """Zpracování jednoho audio chunku"""
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
            wav_file = self.save_audio_to_wav(self.speech_frames)

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
                time.sleep(2)  # Počkej na dokončení transkripce

        finally:
            self.cleanup()

    def cleanup(self):
        """Vyčištění zdrojů"""
        print("🧹 Uklízím zdroje...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# 🚀 Spuštění programu
if __name__ == "__main__":
    vad = VoiceActivityDetector()
    vad.run()