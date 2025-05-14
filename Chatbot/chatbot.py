import pandas as pd
df = pd.read_csv('911_end-to-end_data.csv')
df.head()

df.describe()

df = df.fillna("N/A")

def row_to_text(row):
    return (
        f"On {row['date']}, the {row['agency']} responded to a {row['final_incident_type']} incident. "
        f"The call to first pickup time was {row['call_to_first_pickup']} seconds, "
        f"and the dispatch occurred after {row['call_to_agency_dispatch']} seconds. "
        f"The median dispatch time was {row['median_dispatch']} seconds, "
        f"and the average travel time was {row['average_travel']} seconds."
    )

# df["texts"] = df.apply(row_to_text, axis=1)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
# df["vector"] = df["texts"].apply(lambda x: model.encode(x).tolist())

# !pip install lancedb

# import lancedb
# db = lancedb.connect("chatbot_db")

df.head()
API_KEY = ""
# table = db.create_table("responses", data=df[["texts","vector"]].to_dict(orient="records"))

# query = "What was the average response time for NYPD?"
# query_vec = model.encode(query).tolist()
# results = table.search(query_vec).limit(3).to_df()
# for r in results["texts"]:
#     print(r)


from pathlib import Path
from agno.agent import agent
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini

# df.drop(columns=["texts"], inplace=True)

class AgnoSentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension() # Get dimensions using the correct method

    def get_embedding(self, texts):
        return self.model.encode(texts).tolist()

    def get_embedding_and_usage(self, texts):
        embeddings = self.get_embedding(texts)
        usage = {
            "input_tokens": sum(len(text.split()) for text in texts), 
            "embedding_tokens": 0  
        }
        return embeddings, usage

sentence_transformer_embedder = AgnoSentenceTransformerEmbedder("all-MiniLM-L6-v2")



documents = [row_to_text(row) for _, row in df.iterrows()]

knowlege_base = CSVKnowledgeBase(
    path=Path('.'),
    documents=documents,
    vector_db=LanceDb(
        uri="/content/lancedb",
        table_name="911_data",
        search_type=SearchType.vector,
        embedder=sentence_transformer_embedder,
    ),

)

knowlege_base.load(recreate = False)

from agno.agent import Agent
agent = Agent(
    model = Gemini(id = "gemini-2.5-flash-preview-04-17", api_key = API_KEY),
    knowledge = knowlege_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

def run_cli_chat(agent):
    print("🔹 Welcome to the CSV Chatbot! (type 'exit' to quit)\n")
    while True:
        query = input("🧑‍💻 You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        try:
            # response = agent.get_response(query)
            print(f"🤖 Bot: {agent.print_response(query)}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    run_cli_chat(agent)