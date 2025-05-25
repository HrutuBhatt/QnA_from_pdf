import os, json, time
from pathlib import Path
from agno.agent import agent
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.embedder.openai import OpenAIEmbedder

API_KEY = ""
JSON_FOLDER = "./json_data"
LANCE_DB_URI = "./content/lancedb"
TABLE_NAME = "json_documents"

embedder = OpenAIEmbedder(api_key=API_KEY)
vector_db = LanceDb(
    uri=LANCE_DB_URI,
    table_name=TABLE_NAME,
    search_type=SearchType.vector,
    embedder=embedder
)

# # process rows in batches

def load_json_documents_in_batches(json_folder, batch_size=10, delay_between_batches=20):
    all_docs = []

    #extracting files from mentioned folder
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_docs.extend(data)
                    elif isinstance(data, dict):
                        all_docs.append(data)
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    def batch(iterable, n=1):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]


    #passing rows in batches
    print(f"Total documents to insert: {len(all_docs)}")
    for idx, batch_docs in enumerate(batch(all_docs, batch_size)):
        try:
            texts = [json.dumps(doc) for doc in batch_docs] 
            embeddings = embedder.get_embedding(texts)
            vector_db.upsert(batch_docs,embeddings)
            print(f" Batch {idx + 1} inserted with {len(batch_docs)} documents")
        except Exception as e:
            print(f"Error in batch {idx + 1}: {e}")
        time.sleep(delay_between_batches)  


load_json_documents_in_batches(JSON_FOLDER, batch_size=10)


#knowledge base for agent
knowledge_base = JSONKnowledgeBase(
    path=Path("json_data"),
    vector_db= vector_db
)
knowledge_base.load(recreate = False)


# model 
from agno.agent import Agent
agent = Agent(
    model = Gemini(id = "gemini-2.5-flash-preview-05-20", api_key = API_KEY),
    knowledge = knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)


# CLI for chat
def run_cli_chat(agent):
    print("🔹 Welcome to the CSV Chatbot! (type 'exit' to quit)\n")
    while True:
        query = input("🧑‍💻 You: ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            # response = agent.get_response(query)
            print(f"Bot: {agent.print_response(query)}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    run_cli_chat(agent)