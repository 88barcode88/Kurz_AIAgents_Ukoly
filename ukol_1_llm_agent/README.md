# Praktické cvičení – Lekce 1 [AI Agenti]

## Zadání

Napiš Python skript, který zavolá LLM API, použije nástroj (např. výpočetní funkci) a vrátí odpověď zpět LLM.

## Forma odevzdání

Vypracovaný úkol odevzdejte ve formě zdrojového kódu. Projekt ideálně nahrajte na Github a odevzdejte link do Github repozitáře. Link odevzdejte v Google Classroom.

---

## Jak spustit

1. Nainstaluj závislosti:
   ```
   pip install -r requirements.txt
   ```
2. Vytvoř soubor `.env` do této složky a vlož do něj svůj OpenAI API klíč:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Spusť script:
   ```
   python ukol_1_llm_agent.py
   ```

---

**Deadline:** 10. 7. 2025

---

### Poznámka

Soubor `.env` nesdílej v repozitáři (je v `.gitignore`).  
Použitý model: OpenAI (GPT-4o).  
Výpočetní nástroj: faktoriál čísla.
