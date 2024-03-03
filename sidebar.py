import streamlit as st
import random
 
 
def build_sidebar():
    
    popular_questions = [
            "How can I track my order status on Rakuten?",
            "What is Rakuten's return policy for electronics?",
            "Can I change the shipping address after placing my order?",
            "What payment methods are accepted on Rakuten?",
            "Is it possible to cancel my order after it has been shipped?",
    ]
 
    def insert_as_users_prompt(**kwargs):
        if prompt := kwargs.get("prompt"):
            st.session_state.messages.append({"role": "user", "content": prompt})
 
    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you today?"}
        ]
 
    # App sidebar
    st.image("/mnt/assets/r-chatbot.png", width=50)
    st.write(
        "<h1>Hi, I'm <font color='#ffcdc2'>R-Bot</font> - your personal chat assistant</h1>",
        unsafe_allow_html=True,
    )
    
    st.write(
        "<h2>Ask me anything</h2>",
        unsafe_allow_html=True,
    )
 
    # Pick any 4 questions randomly from popular_questions
    selected_questions = random.sample(popular_questions, 4)
 
    for question in selected_questions:
        st.sidebar.button(
            question,
            on_click=insert_as_users_prompt,
            kwargs={"prompt": question},
            use_container_width=True,
        )
    st.sidebar.markdown("---")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history, type="primary")
