""" Use GPT-3 to classify a bank's website content based on entity, product, and details. """

from typing import List

from pydantic import BaseModel, Field
from langchain import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .retry_parser import retry_parsing

pre_template = """Given the webpage of a bank, your task is to perform an analysis and provide a description of the page's content. Please generate a report with the following sections: 
- Target Audience: Identify and describe the customer group that the service is primarily intended for. 
- Product/Service Description: Detail the primary product or service being offered on the page. 
- Product/Service Details: Provide additional details about the product or service, including its features, benefits, and any other important information that can be found on the page.

Please be brief and use complete sentences and clear, professional language in your analysis.
Also use ENGLISH as the language for your analysis.

{site_content}
"""

pre_prompt = PromptTemplate(
    input_variables=['site_content'],
    template=pre_template,
)

post_template = """ Your task is to categorize the information from a provided analysis of a bank's website page. Please use the following categories in your report:

    Sector: Based on the analysis, identify whether the main audience for the page is 'Individuals' or 'Businesses'.

    Product: From the analysis, determine which of the following bank products or services is primarily featured on the page. If more than one is featured, select the most prominent one. Your options are:
        "Checking Accounts"
        "Savings Accounts"
        "Certificates of Deposit (CDs)"
        "Money Market Accounts"
        "Credit Cards"
        "Debit Cards"
        "Loans"
        "Overdraft Protection"
        "Online and Mobile Banking"
        "Investment Services"
        "Insurance Services"
        "Business Banking Services"

    If none of these categories apply, state 'Not applicable'.

    Details: Based on the analysis, list specific details about the identified product or service. If no specific product or service was identified, provide a general overview of the page content.

    {format_instructions}
    
    Content Analysis: 
    "{content_analysis}"
    
    Answer:
"""


class BankSiteClassification(BaseModel):
    entity: str = Field(description="Sector within the bank")
    product: str = Field(description="Specific service or product")
    details: List[str] = Field(description="List of specific details")


parser = PydanticOutputParser(
    pydantic_object=BankSiteClassification
)

post_prompt = PromptTemplate(
    input_variables=['content_analysis'],
    template=post_template,
    output_parser=parser,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


def classify_site(site_content):
    """ Classify a site based on its content. """
    pre_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=pre_prompt,
        output_key='content_analysis',
    )

    post_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=post_prompt,
        output_key='bank_site_classification',
    )

    bank_site_classification_chain = SequentialChain(
        chains=[pre_chain, post_chain],
        input_variables=['site_content'],
        output_variables=['bank_site_classification'],
    )

    try:
        response = bank_site_classification_chain({'site_content': site_content})['bank_site_classification']
    except Exception as e:
        return {'error': str(e)}

    try:
        parsed_response = retry_parsing(parser, response)
    except Exception as e:
        return {'error': str(e)}

    return parsed_response
