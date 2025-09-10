import whisper      # Knihovna OpenAI pro přepis řeči na text
import pyaudio      # Knihovna pro práci se zvukem (nahrávání z mikrofonu)
import numpy as np  # Numerické operace (v tomto kódu se nevyužívá)
import wave         # Práce s WAV soubory
import tempfile     # Vytváření dočasných souborů
import time         # Časové funkce

# 🔧 Parametry zvuku
RATE = 16000        # Vzorkovací frekvence 16 kHz (standard pro řeč)
CHUNK = 1024        # Velikost bufferu - počet vzorků čtených najednou
RECORD_SECONDS = 3  # Délka každého segmentu nahrávání v sekundách

# 🔽 Načtení Whisper modelu
# Modely podle velikosti: tiny < base < small < medium < large
# Menší = rychlejší, ale méně přesné
model = whisper.load_model("base")

# 🎙️ Inicializace PyAudio pro práci s mikrofonem
p = pyaudio.PyAudio()

# Otevření audio streamu pro nahrávání
stream = p.open(
    format=pyaudio.paInt16,     # 16-bit zvuk
    channels=1,                 # Mono (1 kanál)
    rate=RATE,                  # Vzorkovací frekvence
    input=True,                 # Vstupní stream (nahrávání)
    frames_per_buffer=CHUNK     # Velikost bufferu
)

print("🎙️ Poslouchám... (CTRL+C pro ukončení)")

try:
    # Hlavní smyčka pro kontinuální nahrávání a transkripci
    while True:
        frames = []  # Seznam pro uložení zvukových dat

        # Nahrávání zvuku po chuncích po dobu RECORD_SECONDS
        # POZOR: V originálním kódu je chyba v `RECORD*SECONDS` - mělo by být `RECORD_SECONDS`
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # Čtení dat z mikrofonu
            data = stream.read(CHUNK, exception_on_overflow=False)  # Zabrání chybě při přetížení bufferu
            frames.append(data)  # Přidání dat do seznamu

        # 💾 Vytvoření dočasného WAV souboru
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            # Otevření WAV souboru pro zápis
            wf = wave.open(tmpfile.name, 'wb')
            wf.setnchannels(1)                                    # 1 kanál (mono)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # Šířka vzorku (16-bit)
            wf.setframerate(RATE)                                 # Vzorkovací frekvence
            wf.writeframes(b''.join(frames))                      # Zápis všech nahraných dat
            wf.close()  # Uzavření souboru

            # 🧠 Přepis pomocí Whisper AI modelu
            # OPRAVA: Přidán parametr language="cs" pro češtinu
            result = model.transcribe(tmpfile.name, language="cs")
            print("👂", result["text"])  # Výpis přepsaného textu

        # ⏱️ Krátká pauza před dalším cyklem
        time.sleep(0.1)

except KeyboardInterrupt:
    # Obsluha přerušení uživatelem (Ctrl+C)
    print("🛑 Ukončeno uživatelem.")

# 🧹 Vyčištění zdrojů
stream.stop_stream()  # Zastavení audio streamu
stream.close()        # Uzavření streamu
p.terminate()         # Ukončení PyAudio