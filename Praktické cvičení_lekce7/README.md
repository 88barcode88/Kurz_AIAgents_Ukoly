# LangChain ReAct Agent Demo

ReAct agent postavený na LangChainu. Kombinuje webové vyhledávání (Tavily API) a dotazy do SQLite databáze.

## Požadavky

* Python 3.10+
* `pip`
* Aktivní API klíče pro OpenAI a Tavily

## Instalace

```bash
# 1. Klon repozitáře
git clone https://github.com/88barcode88/Kurz_AIAgents_Ukoly.git
cd Kurz_AIAgents_Ukoly

# 2. Vytvoření virtuálního prostředí
python -m venv venv

# 3. Aktivace venv
#   Windows
venv\Scripts\activate
#   macOS / Linux
source venv/bin/activate

# 4. Instalace balíčků
pip install -r requirements.txt
```

## Konfigurace

V kořeni projektu vytvořte `.env` zkopírováním vzoru:

```bash
cp .env.example .env
```

Otevřete `.env` a doplňte do něj:

```env
OPENAI_API_KEY=sk-...
TAVILI_API_KEY=tvůj_tavili_klíč
```

## Spuštění

```bash
python run_demo.py
```

## Očekávaný výstup

```text
=== LangChain ReAct Agent Demo ===
🔍 Spouštím dotaz přes ReAct agenta:
AI News Results:
Artificial Intelligence ‑ Latest AI News and Analysis ‑ WSJ.com: ...
The latest AI news we announced in June ‑ Google Blog: ...
Database Results for Praha:
Praha je hlavní město České republiky s bohatou historií a kulturou.
```

## Struktura projektu

```text
.
├─ run_demo.py
├─ requirements.txt
├─ .env.example      # vzorový soubor s API klíči
├─ README.md
└─ ...
```

## Licence

MIT
