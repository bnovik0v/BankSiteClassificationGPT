""" This is a demo of bank site classification. """

import os
import requests
from typing import Optional

import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from gpt.bank_site_classification import classify_site as classify_site_gpt
from heuristic.bank_site_classification import classify_site as classify_site_heuristic

load_dotenv()

st.set_page_config(
    page_title="Bank Page Classification",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

st.sidebar.title('Bank Site Classification')
st.sidebar.write('This is a demo of bank site classification. ')
st.sidebar.write('#')

# Ask user to input OPENAI_API_KEY
OPENAI_API_KEY = st.sidebar.text_input('Your OpenAI API key', type='password')
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


class Article(BaseModel):
    title: str = Field(description="Title")
    text: str = Field(description="Text")
    meta: Optional[str] = Field(description="Meta information")

    def __str__(self):
        return f'{self.title}\n\n{self.text}\n\n{self.meta}'

    def __repr__(self):
        return self.__str__()


def get_article(url):
    """ Extract text, title and meta information from a given url. """
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    head = soup.find('head')
    title = head.find('title').text
    meta = head.find('meta', {'name': 'description'})
    if meta:
        meta = meta['content']
    body = soup.find('body')
    main = body.find('main')
    if main:
        body = main
    article = body.find('article')
    if article:
        body = article
    for script in body(["script", "style"]):
        script.decompose()
    text = body.get_text()
    article = Article(title=title, text=text, meta=meta)
    return article


@st.cache_data(ttl=60 * 60 * 24)
def bank_site_classification_gpt(url):
    """ Classify a site based on its content using GPT-3. """
    article = get_article(url)
    article.text = article.text[:10_000]
    result = classify_site_gpt(article)
    return result.dict()


@st.cache_data(ttl=60 * 60 * 24)
def bank_site_classification_heuristic(url):
    """ Classify a site based on its content using heuristics. """
    article = get_article(url)
    result = classify_site_heuristic(str(article))
    return result.dict()


def main():
    st.write('Please, provide the url of the site you want to classify.')
    url = st.text_input('URL', 'https://www.migrosbank.ch/privatpersonen/konten-karten/karten.html')
    st.write('You selected:', url)
    if st.button('Classify', disabled=not OPENAI_API_KEY):
        st.divider()
        col0, col1 = st.columns(2)
        col1.write('Using heuristics:')
        col1.json(bank_site_classification_heuristic(url))
        col0.write('Using GPT:')
        col0.json(bank_site_classification_gpt(url))


if __name__ == '__main__':
    main()
