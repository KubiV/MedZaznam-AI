import os
from groq import Groq

# Nastav klient s API klíčem (předpokládá, že je v environment variable)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cesta k audio souboru na tvém počítači
audio_file_path = "/Users/jakubvavra/Documents/GitHub/Automonitoring-with-AI/tests/vosk/audio files/test_audio.mp3"#"cesta/k/tvemu/audio.mp3"  # Nahraď svou cestou

# Otevři soubor v binárním módu
with open(audio_file_path, "rb") as audio_file:
    # Vytvoř transkripci
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",  # Model pro transkripci
        file=audio_file,  # Audio soubor
        response_format="text"  # Formát výstupu: text, json, verbose_json atd.
    )

# Vypiš transkripci
print(transcription)  # Výstup: Transkribovaný text z audio