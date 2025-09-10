import ollama
import json
from datetime import datetime
import pandas as pd

# --- Krok 1: Systémová instrukce (zůstává stejná) ---
system_prompt = """
Jsi expert na extrakci strukturovaných dat z textu. Tvým úkolem je z textu od uživatele extrahovat názvy potravin a jejich počet. Použij záporná čísla pro vrácené nebo odebrané položky.

Pravidla pro výstup:
1. Výstup musí být VŽDY a POUZE platný JSON objekt.
2. Klíče v JSONu jsou názvy potravin v 1. pádu jednotného čísla (např. "rohlík", "houska", "jablko").
3. Hodnoty jsou seznamy (list) obsahující číselné počty nalezené v textu.

Příklad:
Uživatel pošle: "Mám 2 rohlíky, k tomu pět housek a pak jsem musel vrátit 1 rohlík."
Tvůj výstup bude:
{
  "rohlík": [2, -1],
  "houska": [5]
}
"""

# --- Krok 2: Vstupní text od uživatele ---
user_text = "Mám 2 rohlíky a k tomu dvě housky. Pak jsem dokoupil další 3 rohlíky. Taky jsem koupil 3 bagety a musel jsem kvůli tomu vrátit 2 rohlíky"
print(f"Zpracovávám text: '{user_text}'")

# --- Krok 3: Komunikace s modelem Ollama (zůstává stejný) ---
try:
    response = ollama.chat(
        model='gemma3:1b',  # Ujistěte se, že model máte stažený a spuštěný v Ollama
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_text}
        ],
        options={'response_format': {'type': 'json_object'}}
    )

    # --- Krok 4: Zpracování odpovědi (S VYLEPŠENÍM) ---
    json_output_text = response['message']['content']
    print(f"\nOdpověď od modelu (surový text):\n---\n{json_output_text}\n---")

    # NOVÁ ČÁST: Očištění výstupu od nežádoucích znaků
    # Najdeme první a poslední složenou závorku a vezmeme vše mezi nimi.
    try:
        start_index = json_output_text.find('{')
        end_index = json_output_text.rfind('}') + 1

        if start_index != -1 and end_index != -1:
            clean_json_text = json_output_text[start_index:end_index]
            extracted_data = json.loads(clean_json_text)
            print(f"\nExtrahovaná data (Python slovník):\n{extracted_data}")
        else:
            raise json.JSONDecodeError("JSON objekt nebyl v odpovědi modelu nalezen.", json_output_text, 0)

    except json.JSONDecodeError:
        print("\nChyba: Model vrátil text, ze kterého se nepodařilo extrahovat platný JSON.")
        # Ukončíme další zpracování, protože nemáme data
        extracted_data = None

    # --- Bonus: Zápis dat do tabulky (pokud byla extrakce úspěšná) ---
    if extracted_data:
        records = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for item, counts in extracted_data.items():
            for count in counts:
                records.append({
                    "casova_znamka": timestamp,
                    "polozka": item,
                    "pocet": count
                })

        df = pd.DataFrame(records)
        print("\nVýsledná tabulka pro uložení:")
        print(df)

except Exception as e:
    print(f"\nDošlo k neočekávané chybě: {e}")