


🎤 Mikrofon
   ↓
  ASR (Vosk, Whisper)
   ↓
📄 Text
   ↓
🧠 LLM (Ollama: gemma3, deepseekr1)
   ↓
📦 JSON se strukturou {"": [], ...}
   ↓
Zobrazení ve webovém rozhlaní v reálném čase
   ↓
📈 Uložení do CSV + časová známka




## Instalace Ollama

https://ollama.com/download

Takto nainstalujete model Gemma3 - my jsme zvolili gemma3:4b (4,4 GB)
```ollama run gemma3```

Další možnosti jsou, které budou rychlejší:
gemma3:1b (815MB)
deepseek-r1:1.5b (1.1GB)
Pozor na: https://ollama.com/blog/thinking

Tímto příkazem uvidíte nainstalované modely
```ollama list```

## Python virtual environment¨

1. Vytvoř složku a přejdi do ní
```mkdir mujprojekt && cd mujprojekt```

2. Vytvoř virtuální prostředí
```python3 -m venv venv```

3. Aktivuj prostředí
```source venv/bin/activate```
source venv312/bin/activate

4. Instaluj balíčky
```pip install <nazev_balicku>```

5. Ulož závislosti (volitelné)
```pip freeze > requirements.txt```

6. Deaktivuj prostředí
```deactivate```



# Whisper
https://github.com/openai/whisper

Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
tiny	39 M	tiny.en	tiny	~1 GB	~10x
base	74 M	base.en	base	~1 GB	~7x
small	244 M	small.en	small	~2 GB	~4x
medium	769 M	medium.en	medium	~5 GB	~2x
large	1550 M	N/A	large	~10 GB	1x
turbo	809 M	N/A	turbo	~6 GB	~8x

pip install git+https://github.com/openai/whisper.git
pip install pyaudio numpy

	•	každých 5 sekund (nastavitelné) zaznamená zvuk z mikrofonu
	•	uloží ho jako dočasný .wav soubor ( ~/.cache/whisper/)
	•	přepíše ho pomocí Whisper
	•	vypíše text do konzole

Vylepšení
	•	Detekce, zda někdo mluví (Voice Activity Detection – VAD)
	•	Timestamps (časové značky)

Poznámky:
FP16 vs FP32: Whisper defaultně zkouší použít FP16 (poloviční přesnost) pro rychlejší výpočty


# Vosk
Alternativa k Whisper. Vosk is a speech recognition toolkit https://alphacephei.com/vosk/

pip install vosk

https://alphacephei.com/vosk/models

převést audio na správný formát: ffmpeg -i input_audio.wav -ac 1 -ar 16000 -acodec pcm_s16le converted_audio.wav

# Groq

pip install groq
python3 -m pip install groq --break-system-packages

export GROQ_API_KEY=tvuj-api-klic


# GitIgnore

The git is ignoring these folders:

recordings
logs
venv

So these will be created if using the scripts.