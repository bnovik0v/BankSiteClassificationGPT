""" Use GPT-3 to classify a bank's website content based on entity, product, and details. """

from typing import List

from pydantic import BaseModel, Field
from langchain import OpenAI, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .retry_parser import retry_parsing

template = """Analyze the provided bank website content and categorize the information into: 
    1. Entity: The bank's specific division, e.g., 'Personal Banking', 'Business Banking', 'Investment Banking'. 
    2. Product: The exact service/product offered, e.g., 'Checking Accounts', 'Savings Accounts', 'Credit Cards',
     various types of 'Loans', 'Investment Services', 'Insurance Products'. 
     3. Details: Briefly summarize key features of the service/product.

Your answer should follow these rules:
    a. Each category's response must not exceed four words.
    b. Avoid using product names or the bank's name.
    c. Be short and concise.
    d. Use English.

Website content:
"{site_content}"

{format_instructions}

Answer:
"""


class BankSiteClassification(BaseModel):
    entity: str = Field(description="Sector within the bank")
    product: str = Field(description="Specific service or product")
    details: List[str] = Field(description="List of specific characteristics")


parser = PydanticOutputParser(
    pydantic_object=BankSiteClassification
)

prompt = PromptTemplate(
    input_variables=['site_content'],
    template=template,
    output_parser=parser,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


def classify_site(site_content):
    """ Classify a site based on its content. """
    bank_site_classification_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
    )

    try:
        response = bank_site_classification_chain.predict(site_content=site_content)
    except Exception as e:
        return {'error': str(e)}

    try:
        parsed_response = retry_parsing(parser, response)
    except Exception as e:
        return {'error': str(e)}

    return parsed_response
