# Multi-page-app

# Watch the tutorial video
[How to Make a Multi-Page Web App | Streamlit #16](https://youtu.be/nSw96qUbK9o)


# App URL

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/roapple10/web_app_project/app.py)

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *multipage*
```
conda create -n multipage python=3.7.9
```
Secondly, we will login to the *multipage* environement
```
conda activate multipage
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://github.com/roapple10/Web_app_project/blob/master/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

### Download and unzip this repo

Download [this repo](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/roapple10/Web_app_project) and unzip as your working directory.

###  Launch the app

```
streamlit run app.py
```
