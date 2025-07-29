from dotenv import load_dotenv
import os
import requests
import sqlite3
from typing import Optional, Any

from langchain_openai import OpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine, text

# 1) Načti klíče z .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
tavili_key = os.getenv("TAVILI_API_KEY")

if not openai_key or not tavili_key:
    raise ValueError("Chybí OPENAI_API_KEY nebo TAVILI_API_KEY v .env souboru")

# 2) Definuj nástroj pro Tavili Search
class TaviliSearchTool(BaseTool):
    name: str = "tavili_search"
    description: str = "Hledá na webu přes Tavily API. Vstup je dotaz (string), výstup je text s výsledky."
    
    def _run(self, query: str) -> str:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavili_key, "query": query, "max_results": 5},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "results" in data and data["results"]:
                results = []
                for r in data["results"][:5]:
                    title = r.get("title", "Bez názvu")
                    content = r.get("content", "Bez popisu")
                    results.append(f"{title}: {content}")
                return "\n".join(results)
            else:
                return "Nebyly nalezeny žádné výsledky."
                
        except requests.exceptions.RequestException as e:
            return f"Chyba při volání Tavily API: {str(e)}"
        except Exception as e:
            return f"Neočekávaná chyba: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronní verze není implementována")

# 3) Vytvoř SQLite databázi a tabulku
def setup_database():
    """Vytvoří SQLite databázi a naplní ji testovacími daty"""
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    
    # Vytvoř tabulku docs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            city TEXT
        )
    ''')
    
    # Vložit testovací záznam s Prahou
    cursor.execute('''
        INSERT OR REPLACE INTO docs (id, title, content, city) 
        VALUES (1, 'Dokument o Praze', 'Praha je hlavní město České republiky s bohatou historií a kulturou.', 'Praha')
    ''')
    
    # Přidat další testovací záznamy
    test_data = [
        (2, 'Brno Info', 'Brno je druhé největší město v České republice.', 'Brno'),
        (3, 'Ostrava Guide', 'Ostrava je důležité průmyslové centrum.', 'Ostrava'),
        (4, 'Praha Tourism', 'Praha přitahuje miliony turistů každý rok svými památkami.', 'Praha')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO docs (id, title, content, city) 
        VALUES (?, ?, ?, ?)
    ''', test_data)
    
    conn.commit()
    conn.close()
    print("Databáze data.db byla vytvořena a naplněna testovacími daty.")

# 4) SQL nástroj wrapper
class SQLTool(BaseTool):
    name: str = "sql_database"
    description: str = "Dotazuje se na SQLite databázi s tabulkou 'docs' obsahující sloupce: id, title, content, city. Použij pro hledání informací o městech."
    
    # Definuj privátní atributy pro Pydantic
    _engine: Any = None
    _db: Any = None
    _sql_chain: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Vytvoř SQLAlchemy engine a LangChain SQL utilitu
        self._engine = create_engine("sqlite:///data.db")
        self._db = SQLDatabase(self._engine)
        
        # Vytvoř LLM pro SQL chain
        llm = OpenAI(temperature=0, openai_api_key=openai_key)
        self._sql_chain = SQLDatabaseChain.from_llm(llm, self._db, verbose=True)
    
    def _run(self, query: str) -> str:
        try:
            # Pokud vstup vypadá jako SQL dotaz, spusť ho přímo
            if query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE')):
                with self._engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = result.fetchall()
                    if rows:
                        return "\n".join([str(row) for row in rows])
                    else:
                        return "Dotaz nevrátil žádné výsledky."
            else:
                # Jinak použij LangChain SQL chain pro přirozený jazyk
                return self._sql_chain.run(query)
        except Exception as e:
            return f"Chyba při dotazu do databáze: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronní verze není implementována")

def main():
    # Nastav databázi
    setup_database()
    
    # Inicializuj LLM
    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    
    # Vytvoř nástroje
    tools = [
        TaviliSearchTool(),
        SQLTool()
    ]
    
    # Inicializuj agenta typu ZERO_SHOT_REACT_DESCRIPTION
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    print("=== LangChain ReAct Agent Demo ===")
    print("Agent má k dispozici nástroje: Tavily Search a SQL Database")
    print()
    
    # Příklad spuštění přes agenta
    try:
        print("🔍 Spouštím dotaz přes ReAct agenta:")
        print("-" * 60)
        
        # Použij agenta pro zodpovězení dotazu
        result = agent.invoke({
            "input": "First search for the latest AI news using tavili_search, then query the sql_database for all documents about Praha"
        })
        
        print("-" * 60)
        print("📋 Finální odpověď agenta:")
        print(result.get("output", "Žádný výsledek"))
        
    except Exception as e:
        print(f"❌ Chyba při spuštění agenta: {str(e)}")

if __name__ == "__main__":
    main()