import streamlit as st
from PIL import Image


def app():

    st.sidebar.markdown("""
    Please play around each app page,
    * **IBN_HR_analysis:** My dear..spend time to build
    * **NBA_stats_app:** [Basketball-reference.com](https://www.basketball-reference.com/).
    * **Taskus_Stock_price:** Check my company stock.
    * **SP500:** Check SP500 index.
    * **boston_house:** Kaggle, why we need to do this?
    * **Iris_prediction:** Kaggle, again?
    * **penguins_app:** So cute.
    * **crypto_price_app:** Luna,how are you?
    * **More...**
    """)

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
    -  I learned how to built up the web app by using the Pycharm and git command, I also gained some knowledge of for ML Models deployment.
    -  Building web apps as my side project can hone/sharpen my Python coding skills and Data Science knowledge 
    ''')
    st.info('''
    -  Thanks to [Data Professor](https://www.youtube.com/c/DataProfessor) channel
    -  Thanks to [Python Engineer](https://www.youtube.com/c/PythonEngineer) channel
    ''')


