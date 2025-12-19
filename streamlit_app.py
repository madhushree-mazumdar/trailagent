import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph_app import TrailAgentLangGraph
import time

st.set_page_config(page_title="TrailAgent", layout="wide")

# Add a Home tab at the top of the page with a grey background
st.markdown("""
<nav style="background: #708090; padding: 0.5rem 0 0.5rem 1rem; border-radius: 0 0 12px 12px; margin-bottom: 16px;">
  <a href="/" style="font-weight: bold; color: #F5F5F5; text-decoration: none; font-size: 1.1rem;">🏠 Home</a>
  <a href="/" style="font-weight: bold; color: #F5F5F5; text-decoration: none; font-size: 1.1rem;"> | </a>
  <a href="/" style="font-weight: bold; color: #F5F5F5; text-decoration: none; font-size: 1.1rem;">Future Enhancement</a>
</nav>
""", unsafe_allow_html=True)

# Header with full-width background image
header_image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1400&q=80"

st.markdown(f"""
    <div style="position: relative; width: 100%; height: 280px;">
        <img src='{header_image_url}' style='object-fit: cover; width: 100%; height: 100%; position: absolute; left: 0; top: 0; z-index: 1; border-radius: 0 0 16px 16px;'>
        <div style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.35); z-index: 2; border-radius: 0 0 16px 16px;"></div>
        <div style="position: relative; z-index: 3; display: flex; flex-direction: column; justify-content: center; align-items: flex-start; height: 100%; padding-left: 48px;">
            <h1 style='color: white; margin-bottom: 0;'>TrailAgent</h1>
            <h3 style='color: white; margin-top: 0;'>Your national park exploration guide</h3>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# Main content: Q&A
trailagent_graph = TrailAgentLangGraph()
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''
if 'question' not in st.session_state:
    st.session_state['question'] = ''

# Display the answer above the question input
if st.session_state['answer']:
    st.markdown("### Answer")
    st.write(st.session_state['answer'])
    col_like, col_dislike = st.columns([1, 1])
    with col_like:
        if st.button("👍 Like", key="like_btn"):
            st.success("Thanks for your feedback!")
    with col_dislike:
        if st.button("👎 Dislike", key="dislike_btn"):
            st.info("Sorry the answer wasn't helpful. We'll use your feedback to improve.")
    # Reset the input box to empty after displaying the answer
    st.session_state['question'] = ''

# Question input (empty after answer is generated)
def clear_after_enter():
    # Only process if the input is not empty or whitespace
    q = st.session_state['question'].strip()
    if q:
        st.session_state['answer'] = trailagent_graph.generate_response(q)
        st.session_state['question'] = ''

question = st.text_input("Ask a question about national parks:", key="question", on_change=clear_after_enter)
status_placeholder = st.empty()

st.caption("Built with Streamlit — TrailAgent")

