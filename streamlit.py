import streamlit as st

# Set wide layout and title
st.set_page_config(page_title="My Streamlit Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("🔘 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Description", "🤖 Model", "📊 Dashboard"])

# Description Page
if page == "🏠 Description":
    st.title("📄 Description Page")
    st.write("📝 Write your project description here.")

# Model Page
elif page == "🤖 Model":
    st.title("🤖 Model Page")
    st.write("🧠 Add your model details here.")

# Dashboard Page
elif page == "📊 Dashboard":
    st.title("📊 Dashboard Page")
    st.write("📈 Add charts and data visualizations here.")
