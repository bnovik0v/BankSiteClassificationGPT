""" This is a demo of bank site classification. """

import os
import requests

import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from gpt.bank_site_classification import classify_site

load_dotenv()

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
    meta: str = Field(description="Meta information")

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
    # find main or article tag
    main = body.find('main')
    if main:
        body = main
    article = body.find('article')
    if article:
        body = article
    text = body.get_text()
    article = Article(title=title, text=text, meta=meta)
    return article


def bank_site_classification_gpt(article):
    """ Classify a site based on its content using GPT-3. """
    article.text = article.text[:10_000]
    result = classify_site(article)
    st.markdown('**Entity:**\t' + result.entity)
    st.write('**Product:**\t', result.product)
    st.write('**Details:**\t', result.details)


def main():
    st.write('Please, provide the url of the site you want to classify.')
    url = st.text_input('URL', 'https://www.migrosbank.ch/privatpersonen/konten-karten/karten.html')
    st.write('You selected:', url)
    if st.button('Classify', disabled=not OPENAI_API_KEY):
        st.markdown('---')
        article = get_article(url)
        bank_site_classification_gpt(article)


if __name__ == '__main__':
    main()
