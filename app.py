
import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Groq AI Chatbot")
st.caption("Powered by Groq AI")

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv(override=True)

api_key = None

# First check Streamlit secrets
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

# If not found, check environment variables
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key or not api_key.strip():
    st.error("Groq API key not found!")
    st.info("Add GROQ_API_KEY to your .env file or Streamlit secrets.")
    st.stop()

api_key = api_key.strip()

# -------------------------------
# Initialize Groq Client
# -------------------------------
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Client initialization failed: {e}")
    st.stop()

# -------------------------------
# Model Selection
# -------------------------------
model = st.sidebar.selectbox(
    "Choose AI Model",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
)
# -------------------------------
# System Prompt
# -------------------------------
system_prompt = """
You are a helpful and friendly AI assistant.
Provide clear, accurate, and useful answers.
"""

# -------------------------------
# Initialize Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Clear Chat
# -------------------------------
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -------------------------------
# Display Chat History
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# User Input
# -------------------------------
user_input = st.chat_input("Ask me anything...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ] + st.session_state.messages

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True
            )

            response = ""

            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                response += content

                response_placeholder.markdown(
                    response + "▌"
                )

            response_placeholder.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:

            error_message = str(e)

            if "401" in error_message or "invalid_api_key" in error_message:
                st.error(
                    "Invalid Groq API key. Generate a new key at "
                    "https://console.groq.com/keys"
                )

            elif "429" in error_message:
                st.error(
                    "Groq rate limit reached. Please try again later."
                )

            elif "model_decommissioned" in error_message:
                st.error(
                    "This model is no longer available. Select another model."
                )

            else:
                st.error(f"Groq API Error: {error_message}")

            # Remove failed user message from history
            st.session_state.messages.pop()
