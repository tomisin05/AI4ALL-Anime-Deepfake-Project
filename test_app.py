# import streamlit as st

# st.title("Test App")
# st.write("If you see this, Streamlit is working!")

import streamlit as st

st.title("My Simple Streamlit App")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.write(f"Hello, {name}!")
