import streamlit as st
from multiapp import MultiApp
# import your app modules here
from apps import home, Stock_price,basketball_app,sp500_app,boston_house_ml_app,Iris_prediction,penguins_app,crypto_price_app,IBN_HR_analysis,EDA,excel_merge


st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #EAF0F4 ;">
  <a class="navbar-brand">Ray Lin</a>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="https://github.com/roapple10" target="_blank">Github</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href=".linkedin" target="_blank">Linkedin</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("IBM_HR_analysis", IBN_HR_analysis.app)
app.add_app("NBA_stats_app", basketball_app.app)
app.add_app("Taskus_Stock_price", Stock_price.app)
app.add_app("SP500", sp500_app.app)
app.add_app("boston_house", boston_house_ml_app.app)
app.add_app("Iris_prediction", Iris_prediction.app)
app.add_app("penguins_app", penguins_app.app)
app.add_app("crypto_price_app", crypto_price_app.app)
app.add_app("Excel_merger", excel_merge.app)
app.add_app("EDA Pandas Profiling ", EDA.app)

# The main app
app.run()