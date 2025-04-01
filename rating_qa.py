from google import genai
from google.genai import types
import pandas as pd
import json
df = pd.read_csv("qa_output.csv")
df.head()

'''used to generate rank of answer from 1 to 10 based on its accuracy, from the pdf.'''
def generate():
    client = genai.Client(
        api_key="API_KEY",
    )
    ranks = []
    qa_pairs=""
    batch_size=20


    files = [
        client.files.upload(file="llm_pdf.pdf"),
    ]

    model = "gemini-2.5-pro-exp-03-25"

    '''Divided dataset into batches of 20'''
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]

        qa_pairs=""

        '''for each row in dataset, extract question and answer pairs and append to qa_pairs'''
        for index, row in batch_df.iterrows():
            question = row["Question"]
            answer = row["Answer"]
            qa_pairs += f"Q: {question}\nA: {answer}\n\n"

        '''pass the {qa_pairs} to model'''
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files[0].uri,
                        mime_type=files[0].mime_type,
                    ),
                    types.Part.from_text(text=f"Here are multiple question answer pairs : {qa_pairs}"),
                ],
            ),

        ]

        '''configure the response type (I have used array of ranks corresponding to qa_pairs)'''
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type = types.Type.ARRAY,
                items = types.Schema(
                        type = types.Type.NUMBER,
                ),

            ),
            system_instruction=[
                types.Part.from_text(text="""I'm going to submit multiple question and answer pair. Your task is to evaluate each answer based on the reference PDF and assign a rank between 1 and 10.
                Return the ranks in JSON array format like [9, 7, 5, 10, 8,...]. Please provide exactly one rank per question in the same order. Do not skip any question."""),
            ],
        )

        '''pass the contents and config to model'''
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        '''log error if length of output batch is less than length of input batch size '''
        batch_ranks = json.loads(response.text)
        if len(batch_ranks) != len(batch_df):
                print(f"Warning: Expected {len(batch_df)} ranks, but got {len(batch_ranks)}.")
        print(batch_ranks)

        #append the batch_ranks to ranks
        ranks.extend(batch_ranks)
    # ranks.append(rank)

    '''create new csv file by appending ranks column'''
    df["Rank"]=ranks
    df.to_csv("rank.csv",index=False)


if __name__ == "__main__":
    generate()

