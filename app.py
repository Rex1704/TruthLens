import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model, build_system_prompt
from models.embeddings import get_embedding_model
from utils.rag import process_uploaded_pdf, get_rag_context
from utils.search import get_search_client, get_web_context

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
    st.title("Fact-Checking Chatbot")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    # system_prompt = ""
    
    if "claim_input" not in st.session_state:
        st.session_state["claim_input"] = ""
    
    # # Determine which provider to use based on available API keys
    # chat_model = llm

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
    
    claim_input = st.text_area(
        label="Enter a claim to fact-check:",
        placeholder="e.g. Vaccines cause autism.",
        height=100,
        key="claim_input",
    )

    
    col1, col2 = st.columns(2)
    with col1:
        use_web_search = st.toggle("Use Live Web Search", value=False)
    with col2:
        use_rag = st.toggle("Use Uploaded Documents", value=True)

    check_btn = st.button("Fact Check", type="primary", use_container_width=True)
    
    # Chat input
    # bug -- 'if chat_model:' statement was commented out hence even if the api keys were present it would show 
    # no api keys found until user enter the prompt.
    # if chat_model:
    if check_btn:
        st.markdown(f"**Investigating Claim:** {claim_input}")
        if claim_input is None or claim_input.strip() == "":
            st.warning("Please enter a claim to fact-check.")

        elif not llm:
            st.info(
                "🔧 No API keys found in environment variables. "
                "Please check the Instructions page to set up your API keys."
            )

        else:
            response_mode = st.session_state.get("response_mode", "detailed")

            with st.spinner("Investigating claim..."):

                context_parts = []

                # 1. RAG context from uploaded PDF
                if use_rag and st.session_state.rag_store:
                    with st.status("Searching uploaded documents...", expanded=False):
                        try:
                            rag_ctx = get_rag_context(
                                claim_input, st.session_state.rag_store, embedding_model
                            )
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

                # 3. Combine context and get verdict
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

            # Save to claim history
            st.session_state.history.append({
                "claim": claim_input,
                "verdict": verdict,
                "mode": response_mode,
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
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
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
                "**Concise** - Verdict + one-line reason\n"
                "**Detailed** - Full analysis + evidence breakdown"
            )

            st.divider()

            st.subheader("+ Add Docs")
            uploaded_files = st.file_uploader(
                "Upload a PDF (news article, report, guidelines...)",
                type=["pdf"],
                accept_multiple_files=True,
            )

            if len(uploaded_files) > 0:
                with st.spinner("Processing PDF..."):
                    try:
                        for uploaded_file in uploaded_files:
                            st.session_state.rag_store = process_uploaded_pdf(uploaded_file, embedding_model)
                        st.success(f"'{" ".join(u.name for u in uploaded_files)}' saved.")
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")

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