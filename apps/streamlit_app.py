import streamlit as st
from PIL import Image


#####################
# Header 
st.write('''
# Home Page.
##### *You may find some of the interesting web apps here!* 
''')

image = Image.open('apps/picture/pic.jpg')
st.image(image, width=400)

st.write('[Picture](https://www.freepik.com/vectors/chatbot)')

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
- Experienced Educator, Researcher and Administrator with almost twenty years of experience in data-oriented environment and a passion for delivering insights based on predictive modeling. 
- Strong verbal and written communication skills as demonstrated by extensive participation as invited speaker at `10` conferences as well as publishing 149 research articles.
- Strong track record in scholarly research with H-index of `32` and total citation of 3200+.
''')
