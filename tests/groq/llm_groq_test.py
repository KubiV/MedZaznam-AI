import os
from groq import Groq

# Nastav klient s API klíčem
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Vytvoř chat completion (konverzaci)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Co je to 'zweist'?",  # Tvá zpráva
        }
    ],
    model="llama-3.3-70b-versatile",  # Příklad modelu (můžeš změnit na mixtral-8x7b-32768 nebo jiné dostupné)
    temperature=0.7,  # Volitelně: Kontrola kreativity (0-2)
    max_tokens=150  # Volitelně: Maximální délka odpovědi
)

# Vypiš odpověď modelu
print(chat_completion.choices[0].message.content)