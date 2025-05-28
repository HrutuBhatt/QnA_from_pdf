import os, json, time
from pathlib import Path
from agno.agent import agent
from agno.agent import Agent
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.embedder.openai import OpenAIEmbedder
from agno.media import File

API_KEY = "AIzaSyDg6u-euPuvPvNtJ9lQKxEJWNuI85OuRYo"
JSON_FOLDER = Path("./json_data")


# FILE_KEYWORDS = {
#     "file1.txt": ["nypd", "fdny", "incident type", "dispatch", "arrival", "security", "medical emergencies", "agency", "call", "final_incident_type", "pickup", "dispatch" ,"incidents", "average dispatch", "handoff", "travel", "ems"],
#     "file2.txt": ["ems", "nypd", "agency","description","borough", "incidents" ,"value", "response", "medical", "response time"],
#     "file3.txt": ["stalking", "arrests", "dv", "domestic violence", "arrest", "year"],
#     "file4.txt": ["county", "crime", "type", "anti", "gender", "age", "american", "indian", "alaskan", "white", "black", "protestant", "jewish", "agnoticism", "religious", "buddhist", "greek", "russian", "hispanic", "arab", "disability", "victims", "incidents"],
#     "file5.txt": ["county", "agency", "months", "reported", "crime", "index", "violent", "murder", "robbery", "assault", "burglary", "vehicle", "theft", "region", "property", "year"],
#     "file6.txt": ["county", "felony", "drug", "dwi", "misdemeanor", "property", "total"]
# }

# import re
# from collections import defaultdict

# # from keywords , finds the relevant file
# def get_matching_file(user_query: str) -> str:
#     query = user_query.lower()
#     scores = defaultdict(int)
#     for file, keywords in FILE_KEYWORDS.items():
#         for kw in keywords:
#             if re.search(rf"\b{re.escape(kw.lower())}\b", query):
#                 scores[file] += 1
#     # Return the file with the highest score
#     if scores:
#         best_match = max(scores.items(), key=lambda x: x[1])[0]
#         return best_match
#     else:
#         return None


# agno agent
def build_agent_for_file() -> Agent:

    return Agent(
        model = Gemini(id = "gemini-2.5-flash-preview-05-20", api_key = API_KEY),
        description = "You are a knowledgeable assistant using emergency response data.",
        # knowledge = kb,
        instructions = [
            "Use the JSON data to answer user questions accurately.",
            "If data is not present, respond with 'I could not find relevant data.'"
        ],
        search_knowledge = True,
        show_tool_calls = True,
        markdown = True
    )


agent = build_agent_for_file()

# CLI for chat
def run_cli_chat():
    print("Welcome to the Chatbot! (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # file_name = get_matching_file(query)
        # print(file_name )

        # json_path = Path(f"./json_data/{file_name}")

    

        try:
            print(f"Bot: {agent.print_response(
                query,
                files=[File(filepath=Path(f"./json_data/file1.txt")),
                File(filepath=Path(f"./json_data/file2.txt")),
                File(filepath=Path(f"./json_data/file3.txt")),
                File(filepath=Path(f"./json_data/file4.txt")),
                File(filepath=Path(f"./json_data/file5.txt")),
                File(filepath=Path(f"./json_data/file6.txt"))],
            )}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    run_cli_chat()