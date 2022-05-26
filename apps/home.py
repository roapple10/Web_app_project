import streamlit as st
from PIL import Image


def app():

    #####################
    # Header
    st.write('''
    # Home Page
    ##### *You may find some of the interesting web apps here!* 
    ''')

    image = Image.open('apps/picture/pic.jpg')
    st.image(image, width=450)

    st.write('[Picture source](https://www.freepik.com/vectors/chatbot)')

    st.markdown('## Summary', unsafe_allow_html=True)
    st.info('''
    -  This web app was built on Python script and launched by Streamlit framework
    -  Streamlit is an open-source app framework for Machine Learning and Data Science teams.
    -  I learned how to built up the web app by using the Pycharm and git command, I also gain some knowledge of for ML Models deployment.
    -  Building web apps as my side project can hone/sharpen my Python coding skills and Data Science knowledge 
    ''')


