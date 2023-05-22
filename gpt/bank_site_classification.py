""" Use GPT-3 to classify a bank's website content based on entity, product, and details. """

from pydantic import BaseModel, Field
from langchain import OpenAI, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .retry_parser import retry_parsing

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
        parsed_response = retry_parsing(parser, response, prompt.format_prompt(site_content='...'))
    except Exception as e:
        return {'error': str(e)}

    return parsed_response
