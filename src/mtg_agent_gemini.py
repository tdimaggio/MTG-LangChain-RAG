import os
from pathlib import Path
from typing import List, Optional
# NEW: Load the secret .env file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Setup API Key (Auto-Detected) ---
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    # It will fail gracefully here instead of asking you to paste it

# Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI # <--- NEW IMPORT
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mtg_commander_cards"

# We use Gemini for the brain
LLM_MODEL = "gemini-2.0-flash"
# We KEEP Ollama for the database search (no need to re-index!)
EMBEDDING_MODEL = "nomic-embed-text"

# --- 1. RAG Tool Definition ---

@tool
def card_search_tool(query_str: str) -> str:
    """
    Searches the MTG card database. Input should be a string containing keywords 
    and effects. Example: "Goblin Tribal cards with haste and token generation"
    """
    try:
        # We still use OllamaEmbeddings because that's how we indexed the data!
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        # Retrieve 10 cards for better context
        retrieved_docs = vectorstore.similarity_search(query=query_str, k=10)
    except Exception as e:
        return f"Database Error: {e}"
    
    if not retrieved_docs:
        return "Search Failed: No relevant cards found."
        
    formatted_results = []
    for i, doc in enumerate(retrieved_docs):
        formatted_results.append(f"--- CARD {i+1} ---\n{doc.page_content}\n")
        
    return "The following highly relevant MTG cards were retrieved:\n" + "\n".join(formatted_results)


# --- 2. Initialize the Gemini Agent ---

def initialize_mtg_agent(commander_name: str):
    # 1. LLM (Gemini via Google API)
    # We disable safety settings slightly because MTG text contains "kill", "war", "destroy".
    from langchain_google_genai import HarmBlockThreshold, HarmCategory
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # 2. Tools
    tools = [card_search_tool]
    
    # 3. Prompt Template (ReAct Style)
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**CRITICAL FORMATTING RULES:**
1. Do NOT use the tool like a function. 
   WRONG: Action: card_search_tool("goblin")
   RIGHT: 
   Action: card_search_tool
   Action Input: "goblin"
2. Put "Action Input" on a NEW LINE.

**IMPORTANT GUIDELINES:**
1. **Strategy First:** Determine the best strategy for the Commander yourself.
2. **Search Effectively:** Search for EFFECTS (e.g., "create tokens", "sacrifice outlet") not just names.
3. **Color Safety:** Check the 'ColorIdentity' field in the search results. Do NOT recommend cards that are not allowed in the Commander's color identity.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    # 4. Create Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. Execute
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


# --- 3. Main Execution ---

if __name__ == "__main__":
    print("\n✨ MTG LangChain Commander Agent (Powered by Gemini) ✨")
    print("-------------------------------------------------------")
    
    # 1. Get User Input
    commander = input("Enter your Commander's name: ").strip()
    if not commander:
        commander = "Krenko, Mob Boss"

    # 2. Build Query
    strategy_query = (
        f"My commander is {commander}. "
        "1. Analyze this card and determine the single most effective deck strategy. "
        "2. Use the card_search_tool to find 5 highly synergistic cards for that strategy. "
        "3. Ensure the recommended cards match the Commander's color identity. "
        "List the cards and explain why they fit."
    )
    
    print(f"\n--- Starting Gemini Agent for: {commander} ---")
    
    # 3. Initialize and Run
    mtg_agent = initialize_mtg_agent(commander_name=commander)
    
    try:
        result = mtg_agent.invoke({"input": strategy_query})
        print("\n\n--- 🤖 GEMINI RECOMMENDATION ---")
        print(result['output'])
        print("--------------------------------")
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")