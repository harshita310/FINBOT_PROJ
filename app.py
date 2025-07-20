import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = "vectorstore"

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Helper: Process PDFs and build vectorstore

def process_pdfs():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        return None, "No PDFs found in /data. Please upload PDFs."
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # In-memory Chroma vectorstore (no persist_directory)
    db = Chroma.from_documents(chunks, embedding=embeddings)
    return db, f"Processed {len(documents)} pages, {len(chunks)} chunks."


# Helper: Load vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # In-memory vectorstore (no persist_directory)
    db = Chroma(embedding_function=embeddings)
    return db


# Global: Try to load or build vectorstore
try:
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})
except Exception:
    db, _ = process_pdfs()
    retriever = db.as_retriever(search_kwargs={"k": 3}) if db else None

# LLM
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever) if retriever else None

# Gradio chatbot logic
def chatbot_interface(user_message, history):
    if not user_message.strip():
        user_only = [msg for msg in history if msg['role'] == "user"]
        sidebar_questions = "\n".join(
            f"{idx+1}. {msg['content'].splitlines()[0]}"
            for idx, msg in enumerate(user_only)
        )
        return history, sidebar_questions, ""
    if not qa_chain:
        return history, "", "Vectorstore not loaded. Please upload PDFs."
    # System prompt (not used directly, but can be added to LLM if needed)
    # Append user query
    history.append({"role": "user", "content": user_message})
    # Prepare chat history for qa_chain
    chat_history_pairs = [
        (history[i]["content"], history[i + 1]["content"])
        for i in range(0, len(history) - 1, 2)
    ]
    # Get bot response
    response = qa_chain({
        "question": user_message,
        "chat_history": chat_history_pairs
    })
    bot_reply = response["answer"]
    # Append bot reply
    history.append({"role": "assistant", "content": bot_reply})
    # Sidebar: clean, numbered user questions
    user_only = [msg for msg in history if msg['role'] == "user"]
    sidebar_questions = "\n".join(
        f"{idx+1}. {msg['content'].splitlines()[0]}"
        for idx, msg in enumerate(user_only)
    )
    return history, sidebar_questions, ""

# PDF upload handler
def upload_pdfs(files):
    if not files:
        return "No files uploaded."
    for file in files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    # Rebuild vectorstore
    global db, retriever, qa_chain
    db, msg = process_pdfs()
    retriever = db.as_retriever(search_kwargs={"k": 3}) if db else None
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever) if retriever else None
    return f"PDFs uploaded and processed. {msg}"

# Gradio UI
def build_ui():
    with gr.Blocks(css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500&display=swap');
        body {font-family: 'Inter', sans-serif;}
        .chatbot .user {text-align: right; background: #00BFA6; color: #fff; padding: 10px; border-radius: 15px; margin: 5px;}
        .chatbot .assistant {text-align: left; background: #6C63FF; color: #fff; padding: 10px; border-radius: 15px; margin: 5px;}
        .gradio-container.light {background-color: #FFFFFF; color: #222;}
        .gradio-container.dark {background-color: #1E1E2F; color: #EEE;}
        .chatbot {background-color: inherit; border-radius: 12px; padding: 10px;}
        .sidebar {background-color: inherit; border-radius: 12px; overflow-y: auto;}
        .gr-button {border-radius: 8px; background-color: #6C63FF; color: white;}
        .gr-textbox {border-radius: 8px;}
    """) as app:
        theme_toggle = gr.Checkbox(label="🌗 Dark Mode", value=True)
        with gr.Row():
            with gr.Column(scale=1, min_width=250, elem_classes="sidebar"):
                gr.Markdown("### 📂 Conversation History")
                history_sidebar = gr.Textbox(label="Previous Questions", interactive=False, lines=15)
                gr.Markdown("### 📊 Suggestions")
                suggestion1 = gr.Button("Mutual Funds")
                suggestion2 = gr.Button("Portfolio Diversification")
                suggestion3 = gr.Button("Tax Saving Plans")
                gr.Markdown("### 📄 Upload PDFs")
                pdf_upload = gr.File(file_count="multiple", file_types=[".pdf"])
                upload_btn = gr.Button("Upload & Process PDFs")
                upload_status = gr.Label(value="", label="Upload Status")
            with gr.Column(scale=3, min_width=700):
                gr.HTML("<h1 style='text-align:center; color:#6C63FF;'>🤖 FINBOT Chat</h1>")
                chatbot = gr.Chatbot(label="💬 Chat History", height=500, show_copy_button=True, type="messages", elem_classes="chatbot")
                with gr.Row():
                    user_input = gr.Textbox(placeholder="Ask me anything about finance...", show_label=False, scale=4)
                    send_btn = gr.Button("Send", scale=1)
                status = gr.Label(value="", label="Status")
        # Theme switching logic
        def toggle_theme(is_dark):
            js_code = """
            const container = document.querySelector('.gradio-container');
            if (arguments[0]) {
                container.classList.add('dark');
                container.classList.remove('light');
            } else {
                container.classList.add('light');
                container.classList.remove('dark');
            }
            """
            return gr.HTML(f"<script>{js_code}</script>")
        theme_toggle.change(toggle_theme, inputs=theme_toggle, outputs=None)
        history_state = gr.State([])
        # Clickable suggestions
        suggestion1.click(lambda: "Tell me about Mutual Funds", outputs=user_input)
        suggestion2.click(lambda: "How to diversify my portfolio?", outputs=user_input)
        suggestion3.click(lambda: "What are the best tax saving plans?", outputs=user_input)
        # Send button logic
        send_btn.click(
            chatbot_interface,
            inputs=[user_input, history_state],
            outputs=[chatbot, history_sidebar, user_input],
            show_progress="full"
        ).then(
            lambda: "✍️ FINBOT is typing...",
            None, status
        ).then(
            lambda: "", None, status
        )
        # PDF upload logic
        upload_btn.click(
            upload_pdfs,
            inputs=pdf_upload,
            outputs=upload_status
        )
    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(share=True) 