import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model, build_system_prompt, is_fact_check_input
from models.embeddings import get_embedding_model
from utils.rag import process_uploaded_pdf, get_rag_context
from utils.search import get_search_client, get_web_context
from utils.tweet import is_tweet_url, fetch_tweet_text

@st.cache_resource
def load_llm_model():
    try:
        return get_chatgroq_model()
    except Exception as e:
        st.error(f"Error loading LLM model: {str(e)}")
        return None
    
@st.cache_resource
def load_embedding_model():
    try:
        return get_embedding_model()
    except Exception as e:
        st.error(f"Embedding model init failed: {e}")
        return None
    
@st.cache_resource
def load_search_client():
    try:
        return get_search_client()
    except Exception as e:
        st.warning(f"Web search unavailable: {e}")
        return None


def get_chat_response(chat_model, claim: str, context: str, response_mode: str) -> str:
    try:
        system_prompt = build_system_prompt(response_mode)

        formatted_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Please fact-check the following claim:\n\n"
                    f"CLAIM: {claim}\n\n"
                    f"CONTEXT FROM SOURCES:\n"
                    f"{context if context else 'No additional context available. Use your general knowledge.'}"
                )
            ),
        ]

        response = chat_model.invoke(formatted_messages)
        return response.content

    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)




def chat_page(llm, embedding_model, search_client):
    """Main chat interface page"""
    st.title("TruthLens - AI Fact-Checking Assistant")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    # system_prompt = ""
    
    if "claim_input" not in st.session_state:
        st.session_state["claim_input"] = ""

    if "tweet_cache" not in st.session_state:
        st.session_state["tweet_cache"] = {"url": None, "result": None}
    
    # # Determine which provider to use based on available API keys
    # chat_model = llm

    def on_claim_change():
        value = st.session_state["claim_input"].strip()
        cache = st.session_state["tweet_cache"]
 
        if is_tweet_url(value):
            if cache["url"] != value:
                result = fetch_tweet_text(value)
                st.session_state["tweet_cache"] = {"url": value, "result": result}
        else:
            st.session_state["tweet_cache"] = {"url": None, "result": None}
 

    st.markdown("**Try an example:**")
    examples = [
        "The Great Wall of China is visible from space.",
        "5G towers spread COVID-19.",
        "Drinking 8 glasses of water a day is scientifically required.",
    ]
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i].button(f'"{example[:38]}..."', key=f"ex_{i}"):
            st.session_state["claim_input"] = example
            st.session_state["tweet_cache"] = {"url": None, "result": None}

    
    claim_input = st.text_area(
        label="Enter a claim to fact-check:",
        placeholder="e.g. Vaccines cause autism.",
        height=100,
        key="claim_input",
        on_change=on_claim_change,
    )
    
    tweet_context = None
    tweet_author = None
 
    cached = st.session_state["tweet_cache"]
    if cached["url"] and cached["result"]:
        result = cached["result"]
        if result["success"]:
            tweet_author = result["author"]
            tweet_context = result["text"]
            claim_input = f"{claim_input}\n\n[TWEET BY @{tweet_author}]: {tweet_context}"
 
            st.markdown(
                f"""
                <div style="
                    background: #e8f4fd;
                    border-left: 4px solid #1da1f2;
                    border-radius: 6px;
                    padding: 12px 16px;
                    margin: 4px 0 12px 0;
                    font-size: 14px;
                    color: #1a1a1a;
                    line-height: 1.5;
                ">
                    <span style="color:#1da1f2; font-weight:600;">@{tweet_author}</span>
                    <br><br>
                    {tweet_context}
                    <br><br>
                    <span style="color:#888; font-size:12px;">Source: {cached["url"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Could not fetch tweet: {result['error']}")
    else:
        st.markdown(""" """)

    with st.expander("Attach a reference document", expanded=False):
        st.caption("Upload a PDF relevant to this specific claim - e.g. a report, news article, or research paper.")
        uploaded_files = st.file_uploader(
            label="Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="inline_pdf_upload",
            label_visibility="collapsed",
        )
    
    col1, col2 = st.columns(2)
    with col1:
        use_web_search = st.toggle("Use Live Web Search", value=False)
    with col2:
        use_rag = st.toggle(
            "Use Uploaded Documents",
            value=bool(uploaded_files),
            disabled=not bool(uploaded_files),
        )

    check_btn = st.button("Fact Check", type="primary", use_container_width=True)
    
    # Chat input
    # bug -- 'if chat_model:' statement was commented out hence even if the api keys were present it would show 
    # no api keys found until user enter the prompt.
    # if chat_model:
    if check_btn:
        # st.markdown(f"**Investigating Claim:** {claim_input}")
        if claim_input is None or claim_input.strip() == "":
            st.warning("Please enter a claim to fact-check.")

        elif not llm:
            st.info(
                "🔧 No API keys found in environment variables. "
                "Please check the Instructions page to set up your API keys."
            )

        else:
            is_valid, rejection_reason = is_fact_check_input(claim_input)
            if not is_valid:
                st.warning(rejection_reason)
                st.stop()
            response_mode = st.session_state.get("response_mode", "detailed")

            with st.spinner("Investigating claim..."):

                context_parts = []

                local_rag_store = []
                if use_rag and uploaded_files:
                    with st.status("Processing uploaded documents...", expanded=False):
                        try:
                            for uploaded_file in uploaded_files:
                                local_rag_store.append(process_uploaded_pdf(uploaded_file, embedding_model))
                            st.write(f"Loaded {len(uploaded_files)} file(s).")
                        except Exception as e:
                            st.warning(f"PDF processing failed: {e}")
 
                if local_rag_store:
                    with st.status("Searching uploaded documents...", expanded=False):
                        try:
                            rag_ctx = get_rag_context(claim_input, local_rag_store, embedding_model)
                            context_parts.append("=== FROM UPLOADED DOCUMENTS ===\n" + rag_ctx)
                        except Exception as e:
                            st.warning(f"RAG search failed: {e}")
                        st.write("Done.")
                
                if use_web_search and search_client:
                    with st.status("Searching the web...", expanded=False):
                        try:
                            web_ctx = get_web_context(search_client, claim_input)
                            context_parts.append("=== FROM WEB SOURCES ===\n" + web_ctx)
                        except Exception as e:
                            st.warning(f"Web search failed: {e}")
                        st.write("Done.")

                full_context = "\n\n".join(context_parts)
                verdict = get_chat_response(llm, claim_input, full_context, response_mode)

            st.markdown("---")
            st.subheader("Fact-Check Result")

            if "TRUE" in verdict.upper():
                st.success(verdict)
            elif "FALSE" in verdict.upper():
                st.error(verdict)
            elif "MISLEADING" in verdict.upper():
                st.warning(verdict)
            else:
                st.info(verdict)

            sources_used = []
            if local_rag_store:
                sources_used.append(f"{len(uploaded_files)} doc(s)")
            if use_web_search and search_client:
                sources_used.append("Web")
            if sources_used:
                st.caption(f"Sources used: {' · '.join(sources_used)} | Mode: {response_mode.capitalize()}")

            st.session_state.history.append({
                "claim": claim_input,
                "verdict": verdict,
                "mode": response_mode,
                "docs": [f.name for f in uploaded_files] if uploaded_files else [],
            })

def main():
    embedding_model = load_embedding_model()
    llm = load_llm_model()
    search_client = load_search_client()

    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None

    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.set_page_config(
        page_title="TruthLens - AI Fact-Checker",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()

            st.subheader("Response Mode")
            response_mode = st.radio(
                label="Select mode:",
                options=["concise", "detailed"],
                format_func=lambda x: "Concise" if x == "concise" else "Detailed",
                index=1,
            )
            st.session_state["response_mode"] = response_mode
            st.caption(
                "**Concise** - Verdict + one-line reason\n\n"
                "**Detailed** - Full analysis + evidence breakdown"
            )

            st.divider()

            st.subheader("Claim History")
            if st.session_state.history:
                for i, item in enumerate(reversed(st.session_state.history[-10:])):
                    label = item["claim"][:35] + "..." if len(item["claim"]) > 35 else item["claim"]
                    with st.expander(label):
                        st.markdown(f"**Mode:** {item['mode'].capitalize()}")
                        st.markdown(item["verdict"])
            else:
                st.caption("No claims checked yet.")

            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.history = []
                st.session_state.messages = []
                st.rerun()

            
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page(llm, embedding_model, search_client)

if __name__ == "__main__":
    main()