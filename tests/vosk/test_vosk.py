import wave
import json
from vosk import Model, KaldiRecognizer
import sys

# Cesta k WAV souboru a modelu Vosk
audio_file = "/Users/jakubvavra/Desktop/Automoitoring/tests/vosk/audio files/test_audio_converted.wav"
model_path = "/Users/jakubvavra/Desktop/Automoitoring/tests/vosk/vosk-model-small-cs-0.4-rhasspy"#"vosk-model-cs"  # Upravte podle cesty k vašemu modelu

# Kontrola, zda je soubor WAV mono a má správný formát
def check_wav_file(wav_file):
    with wave.open(wav_file, "rb") as wf:
        if wf.getnchannels() != 1:
            print("Chyba: Audio soubor musí být mono!")
            sys.exit(1)
        if wf.getsampwidth() != 2:
            print("Chyba: Audio soubor musí mít 16-bitový formát!")
            sys.exit(1)
        if wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
            print("Chyba: Nepodporovaná vzorkovací frekvence!")
            sys.exit(1)

# Funkce pro transkripci audia
def transcribe_audio(audio_path, model_path):
    # Načtení modelu Vosk
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)  # Upravte vzorkovací frekvenci podle vašeho WAV souboru

    # Otevření WAV souboru
    with wave.open(audio_path, "rb") as wf:
        check_wav_file(audio_path)
        # Čtení dat po částech
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                if "text" in result_dict and result_dict["text"]:
                    print(result_dict["text"])
                    with open("transcription.txt", "a", encoding="utf-8") as f:
                        f.write(result_dict["text"] + "\n")

        # Získání finálního výsledku
        final_result = recognizer.FinalResult()
        result_dict = json.loads(final_result)
        if "text" in result_dict and result_dict["text"]:
            print(result_dict["text"])
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(result_dict["text"] + "\n")

if __name__ == "__main__":
    try:
        transcribe_audio(audio_file, model_path)
        print("Transkripce dokončena, výsledek uložen do transcription.txt")
    except Exception as e:
        print(f"Chyba při transkripci: {str(e)}")