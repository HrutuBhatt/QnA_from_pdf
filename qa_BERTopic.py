
# Using Topic Modelling
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from pypdf import PdfReader
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
import json

text=""

'''Extract texts from pdf'''
with open("llm_pdf.pdf", "rb") as file:
    reader = PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

'''Create bertopic model , KeyBERTInspired representation model is used'''
topic_model = BERTopic(representation_model=KeyBERTInspired())
topics, probs = topic_model.fit_transform(text.split("\n"))

topic_model.get_topic_info()
# topic_model.get_topic(0)

#Visualization
fig1 = topic_model.visualize_barchart(top_n_topics=100)
fig1.show(renderer="colab")

fig2 = topic_model.visualize_topics()
fig2.show(renderer="colab")

topic_model.visualize_heatmap()

fig3 = topic_model.visualize_distribution(probs)
fig3.show(renderer="colab")

topic_model.reduce_topics(text.split("\n"), nr_topics=100)
topics = topic_model.topics_

topic_model.get_topic_info()

'''assigning topics to questions from pdf'''
import pandas as pd
df = pd.read_csv('qa_output.csv')
list_of_questions = df['Question'].tolist()
topic_assignments, _ = topic_model.transform(list_of_questions)
df["Topic"] = topic_assignments

'''to check how much parts of topics are covered , check the topics of questions previously generated.'''
import seaborn as sns
import matplotlib.pyplot as plt

topic_counts = df["Topic"].value_counts().sort_index()
sns.heatmap([topic_counts.values], cmap="Blues", annot=True, xticklabels=topic_counts.index)
plt.xlabel("Topic Number")
plt.ylabel("Coverage")
plt.title("Topic Distribution Across Questions")
plt.show()

#getting list of those topics
topics = list(set(topic_model.topics_))
topics = [t for t in topics if t != -1]
topic_labels = topic_model.get_topic_info()
topics_list = topic_labels["Name"].tolist()

print(topics_list)

#some topics are underrepresented from above graph
'''function to generate Q/A from the list of topics provided'''

def generate():
    client = genai.Client(
        api_key="API_KEY",
    )

    files = [
        client.files.upload(file="llm_pdf.pdf"),
    ]

    model = "gemini-2.5-pro-exp-03-25"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=f"Here are the topics that need to be covered from the pdf :{topics_list}"),
            ],
        ),

    ]

    '''configure the response type '''
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.ARRAY,
            items=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={
                    "question": genai.types.Schema(type=genai.types.Type.STRING),
                    "answer": genai.types.Schema(type=genai.types.Type.STRING),
                    "topic": genai.types.Schema(type=genai.types.Type.STRING),
                },
            ),
        ),
        system_instruction=[
            types.Part.from_text(text="""Generate 200 different question-answer pairs based on topics provided , from pdf.
            Ensure that whole pdf should be covered to generate question, and covers all topics evenly. Questions should be relevant having precise answers.
            The output should be in JSON format with the following structure:


            [
                {"question": "What is reinforcement learning?", "answer": "It is a type of ML where agents learn by trial and error.", "topic": "Machine Learning"},...
            ] """
            ),
        ],
    )

    '''pass the contents and config to model'''
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    print(response.text)
    try:
        qa_pairs = response.text  # Extract JSON string
        qa_pairs = eval(qa_pairs)  # Convert JSON string to Python list of dicts
    except Exception as e:
        print(f"Error parsing response: {e}")
        return

    df = pd.DataFrame(qa_pairs)
    df.to_csv("extracted_questions.csv", index=False)


if __name__ == "__main__":
    generate()

# check how much topic are covered by the new prompt. Plot the 'topics number' and its corresponding questions generated in graph,
df2 = pd.read_csv('extracted_questions.csv')
list_of_questions = df2['question'].tolist()
topic_assignments, _ = topic_model.transform(list_of_questions)
df2["Topic_predicted"] = topic_assignments

topic_counts = df2["Topic_predicted"].value_counts().sort_index()
sns.heatmap([topic_counts.values], cmap="Blues", annot=True, xticklabels=topic_counts.index)
plt.xlabel("Topic Number")
plt.ylabel("Coverage")
plt.title("Topic Distribution Across Questions")
plt.show()


