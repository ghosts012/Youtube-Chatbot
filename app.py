import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration and Keys
load_dotenv()
geminiApiKey = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("📺 YouTube Video Chatbot")

# 2. Sidebar Logic
with st.sidebar:
    youtubeUrl = st.text_input("Enter YouTube URL:")
    processBtn = st.button("Process Video")

# Function to build vector store with camelCase naming
@st.cache_resource
def prepareVectorStore(videoUrl):
    try:
        videoId = videoUrl.split("v=")[1].split("&")[0]
        api = YouTubeTranscriptApi()
        transcriptData = api.fetch(videoId, languages=['en', 'hi'])
        fullTranscript = " ".join([snippet.text for snippet in transcriptData])
        
        textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textChunks = textSplitter.create_documents([fullTranscript])
        
        huggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorStore = FAISS.from_documents(textChunks, huggingFaceEmbeddings)
        return vectorStore
    except Exception as error:
        st.error(f"Error processing video: {error}")
        return None

# 3. Processing Trigger
if youtubeUrl and processBtn:
    with st.spinner("Analyzing video..."):
        st.session_state.vectorStore = prepareVectorStore(youtubeUrl)
        st.success("Analysis complete! Ask your questions below.")

# 4. Chat Interface Logic
if "vectorStore" in st.session_state:
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # Display previous messages
    for chatMessage in st.session_state.chatHistory:
        with st.chat_message(chatMessage["role"]):
            st.markdown(chatMessage["content"])

    # User Input Field
    if userInput := st.chat_input("Ask about the video..."):
        st.session_state.chatHistory.append({"role": "user", "content": userInput})
        with st.chat_message("user"):
            st.markdown(userInput)

        # Build RAG Chain
        retriever = st.session_state.vectorStore.as_retriever(search_kwargs={"k": 4})
        modelLlm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=geminiApiKey)
        
        ragPromptTemplate = """
        Answer the question using ONLY the provided transcript context. 
        If the answer is not in the context, say "I don't know".
        
        Context: {context}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(ragPromptTemplate)
        
        def formatDocuments(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain Definition
        mainChain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(formatDocuments),
                "question": RunnablePassthrough()
            })
            | prompt
            | modelLlm
            | StrOutputParser()
        )

        # Generate and display response
        with st.chat_message("assistant"):
            botResponse = mainChain.invoke(userInput)
            st.markdown(botResponse)
            st.session_state.chatHistory.append({"role": "assistant", "content": botResponse})