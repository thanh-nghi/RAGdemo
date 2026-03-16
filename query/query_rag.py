from rag.rag_chatbot import ask_rag
print("RAG Chatbot Ready")
while True:
    query = input("\nAsk: ")
    if query.lower() == "exit":
        break
    answer = ask_rag(query)

    print(f"Answer: {answer}")
    