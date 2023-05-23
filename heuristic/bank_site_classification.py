""" Bank site classification module. """

from typing import List
from collections import defaultdict, Counter
import json

from pydantic import Field, BaseModel
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

with open('heuristic/keywords.json') as f:
    financial_keywords = json.load(f)


class BankSiteClassification(BaseModel):
    entity: str = Field(description="Sector within the bank")
    product: str = Field(description="Specific service or product")
    details: List[str] = Field(description="Specific characteristic")


def extract_keywords(text):
    """ Extract keywords from a given text. """
    text = text.lower()
    return [t for t in word_tokenize(text) if t in financial_keywords]


def keywords_to_topic(keywords):
    """ Convert keywords to topic. """
    topics = defaultdict(int)
    keywords_count = Counter(keywords)
    for keyword, count in keywords_count.items():
        related_topics = financial_keywords[keyword]
        for topic in related_topics:
            topics[topic] += count
    topic = Counter(topics).most_common()
    if topic:
        return topic[0][0]
    else:
        return None


def classify_site(site_content):
    """ Classify a given site content. """
    keywords = extract_keywords(site_content)
    if keywords:
        site_topic = keywords_to_topic(keywords)
        keywords = list({kw for kw in keywords if site_topic in financial_keywords[kw]})
        return BankSiteClassification(entity='Private', product=site_topic, details=keywords)
    else:
        return BankSiteClassification(entity='Private', product='Unknown', details=['Unknown'])
