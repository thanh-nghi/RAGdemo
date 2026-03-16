from rag.retriever import retrieve
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))

def ask_rag(question):
    context = retrieve(question)
    context_text = '\n\n'.join(context)
    prompt = f"""
Use the following context to answer the question.
Context:
{context_text}
Question: {question}
Answer: """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text