from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def build_rag_chain(llm, retriever, chat_history):
    # Rephrasing prompt
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a helpful assistant.\n"
        "Given a conversation and a follow-up question, rephrase the question into a self-contained standalone query.\n"
        "Only rephrase if necessary. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Final answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an assistant. Use the following context to answer the question clearly. \n"
        "Only use the given context. If the answer isn't in the context, say \"I'm not sure based on the information provided.\"\n"
        "Keep the answer in 3 to 4 sentences unless told otherwise.\n\n"
        "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create memory-aware retriever
    hist_aware = create_history_aware_retriever(llm, retriever, context_prompt)
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag = create_retrieval_chain(hist_aware, answer_chain)

    # Chat history manager — only one instance for terminal
    def get_history(_: str):  # session_id ignored
        return chat_history

    return RunnableWithMessageHistory(
        rag,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
