""" This is a demo of bank site classification. """

import os
import requests
import streamlit as st
from langchain import OpenAI, LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()


st.sidebar.title('Bank Site Classification')
st.sidebar.write('This is a demo of bank site classification. ')
st.sidebar.write('#')

# Ask user to input OPENAI_API_KEY
OPENAI_API_KEY = st.sidebar.text_input('Your OpenAI API key', type='password')
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

template = """Analyze the bank's website content and categorize it based on:

    Entity: Identify and distinguish the sector within the bank, such as 'Personal Banking', 
    'Business Banking', 'Investment Banking', among others.

    Product: Ascertain the specific service or product offered by the bank. This can be 'Checking Accounts', 
    'Savings Accounts', 'Credit Cards', various types of 'Loans' (like 'Mortgage Loans', 'Personal Loans', 
    'Auto Loans'), 'Investment Services', 'Insurance Products', and more.

    Details: Briefly highlight the specific characteristic or feature of the aforementioned product or service. This 
    could involve details like interest rates, fees, benefits, terms and conditions, and other pertinent information.

{format_instructions}

REMEMBER: No more than 4 words for each variable. Don't use names of products or bank in your answer.

Site content:
"{site_content}"

Answer:
"""


class BankSiteClassification(BaseModel):
    entity: str = Field(description="Sector within the bank")
    product: str = Field(description="Specific service or product")
    details: str = Field(description="Specific characteristic")


parser = PydanticOutputParser(
    pydantic_object=BankSiteClassification
)

prompt = PromptTemplate(
    input_variables=['site_content'],
    template=template,
    output_parser=parser,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

bank_site_classification_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
)


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


def main():
    st.write('Please, provide the url of the site you want to classify.')
    # user can enter or select an url from list
    url = st.text_input('URL', 'https://www.migrosbank.ch/privatpersonen/konten-karten/karten.html')
    st.write('You selected:', url)
    if st.button('Classify', disabled=not OPENAI_API_KEY):
        st.markdown('---')
        article = get_article(url)
        article.text = article.text[:10_000]
        response = bank_site_classification_chain.predict(site_content=str(article))
        result = parser.parse(response)
        st.markdown('**Entity:**\t' + result.entity)
        st.write('**Product:**\t', result.product)
        st.write('**Details:**\t', result.details)


if __name__ == '__main__':
    main()
