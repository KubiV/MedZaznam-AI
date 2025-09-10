import wave
import json
import os
from vosk import Model, KaldiRecognizer

# Cesta k WAV souboru a modelu
audio_file = "/Users/jakubvavra/Desktop/Automoitoring/tests/vosk/audio files/ahoj_test_converted.wav"
model_path = "/Users/jakubvavra/Desktop/Automoitoring/tests/vosk/vosk-model-small-cs-0.4-rhasspy"#"vosk-model-cs"  # Upravte podle cesty k vašemu modelu

try:
    # Kontrola existence složky modelu
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Složka modelu {model_path} neexistuje!")

    # Načtení modelu
    print(f"Načítám model z: {model_path}")
    model = Model(model_path)

    # Otevření WAV souboru a kontrola formátu
    with wave.open(audio_file, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Audio musí být mono!")
        if wf.getsampwidth() != 2:
            raise ValueError("Audio musí být 16-bit!")
        sample_rate = wf.getframerate()
        print(f"Vzorkovací frekvence audia: {sample_rate} Hz")

        # Inicializace recognizeru s dynamickou vzorkovací frekvencí
        recognizer = KaldiRecognizer(model, sample_rate)

        # Transkripce po částech
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if "text" in result and result["text"]:
                    print(f"Výsledek: {result['text']}")
                    with open("transcription.txt", "a", encoding="utf-8") as f:
                        f.write(result["text"] + "\n")
            else:
                partial = json.loads(recognizer.PartialResult())
                if "partial" in partial and partial["partial"]:
                    print(f"Částečný výsledek: {partial['partial']}")

        # Finální výsledek
        final_result = json.loads(recognizer.FinalResult())
        if "text" in final_result and final_result["text"]:
            print(f"Finální výsledek: {final_result['text']}")
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(final_result["text"] + "\n")

    print("Transkripce dokončena, výsledek uložen do transcription.txt")

except FileNotFoundError as e:
    print(f"Chyba: {str(e)}")
except Exception as e:
    print(f"Chyba při transkripci: {str(e)}")