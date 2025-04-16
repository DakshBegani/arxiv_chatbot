import os
import streamlit as st
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Hardcoded token for now
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="🧠 arXiv Chatbot", page_icon="📚")
st.title("📚 arXiv Research Chatbot")

# Step 1: Topic input
if "vectorstore" not in st.session_state:
    topic = st.text_input("Enter a research topic to fetch relevant papers:")
    if topic:
        with st.spinner("🔍 Scraping arXiv..."):
            search = arxiv.Search(query=topic, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
            papers = []
            for result in search.results():
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d")
                }
                papers.append(paper)
            st.session_state["papers"] = papers

        with st.spinner("⚙️ Creating vectorstore..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docs = []
            for paper in st.session_state["papers"]:
                text = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nDate: {paper['published']}\n\n{paper['summary']}"
                docs.append(Document(page_content=text))
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state["vectorstore"] = vectorstore

        with st.spinner("🤖 Loading language model..."):
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
            st.session_state["llm"] = HuggingFacePipeline(pipeline=pipe)

        retriever = st.session_state["vectorstore"].as_retriever()
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=st.session_state["llm"],
            retriever=retriever
        )

        st.success("✅ Chatbot ready! Ask questions below.")

# Step 2: Chat UI
if "qa_chain" in st.session_state:
    st.divider()
    st.subheader("💬 Chat with the papers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a research question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state["qa_chain"].run(user_input)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
