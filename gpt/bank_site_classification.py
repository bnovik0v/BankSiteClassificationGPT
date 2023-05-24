""" Use GPT-3 to classify a bank's website content based on entity, product, and details. """

from typing import List

from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import (
    SystemMessage
)

from .retry_parser import retry_parsing
from .usage_info import return_usage_info

human_message_template = """Given the webpage of a bank, your task is to perform an analysis and provide a description of the page's content. Please generate a brief report with the following sections: 
- Target Audience: Identify and describe the customer group that the service is primarily intended for: individuals or business. 
- Product/Service Description: Detail the primary product or service being offered on the page. 
- Product/Service Details: Provide additional details about the product or service, including its features, benefits, and any other important information that can be found on the page.

IMPORTANT: Be brief (no more than 3 sentences for each answer). Use complete sentences and clear, professional language in your analysis.
Also use ENGLISH as the language for your analysis.

{site_content}
"""

human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)
system_message_prompt = SystemMessage(content="You are a bank employee analyzing a bank's website page.")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

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

    Details: Based on the analysis, list brief and shortly specific details about the identified product or service.

    {format_instructions}
    
    Content Analysis: 
    "{content_analysis}"
    
    Answer:
"""


class BankSiteClassification(BaseModel):
    sector: str = Field(description="Sector within the bank")
    product: str = Field(description="Specific service or product")
    details: List[str] = Field(description="List of specific details")


parser = PydanticOutputParser(
    pydantic_object=BankSiteClassification
)

human_message_prompt2 = HumanMessagePromptTemplate.from_template(post_template)
chat_prompt2 = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt2])


def ask_chatgpt(messages):
    """ Ask chatgpt for a response. """
    chat = ChatOpenAI(
        temperature=0,
    )

    return chat(messages).content


#@return_usage_info
def classify_site(site_content):
    """ Classify a site based on its content. """
    try:
        content_analysis = ask_chatgpt(chat_prompt.format_prompt(site_content=site_content).to_messages())
    except Exception as e:
        return {'error': str(e)}

    try:
        content_classification = ask_chatgpt(
            chat_prompt2.format_prompt(
                content_analysis=content_analysis,
                format_instructions=parser.get_format_instructions()
            ).to_messages()
        )
    except Exception as e:
        return {'error': str(e)}

    try:
        parsed_response = retry_parsing(parser, content_classification)
    except Exception as e:
        return {'error': str(e)}

    return parsed_response
