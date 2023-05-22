from __future__ import annotations

from typing import Any

from langchain import OpenAI
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    BaseOutputParser,
    OutputParserException,
)

NAIVE_COMPLETION_RETRY = """Json:
```
{json}
```

Above, the Json is not valid.

{format_instructions}

Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)


class RetryOutputParserForJson(BaseOutputParser):
    """Wraps a parser and tries to fix parsing errors for json.

    Does this by passing the original prompt and the completion to another
    LLM, and telling it the completion did not satisfy criteria in the prompt.
    """

    parser: BaseOutputParser
    retry_chain: LLMChain

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            parser: BaseOutputParser,
            prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
    ) -> RetryOutputParserForJson:
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain)

    def parse_with_prompt(self, completion: str, format_instructions: str = '') -> Any:
        try:
            parsed_completion = self.parser.parse(completion)
        except OutputParserException:
            new_completion = self.retry_chain.run(
                json=completion, format_instructions=format_instructions
            )
            parsed_completion = self.parser.parse(new_completion)

        return parsed_completion

    def parse(self, completion: str) -> Any:
        raise NotImplementedError(
            "This OutputParser can only be called by the `parse_with_prompt` method."
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()


def retry_parsing(parser, bad_response):
    """
    Retry parsing the response with RetryWithErrorOutputParser.
    :param parser: parser to use
    :param bad_response: bad response
    :param prompt_value: prompt value
    :return: fixed response
    """
    retry_parser = RetryOutputParserForJson.from_llm(
        parser=parser,
        llm=OpenAI(temperature=0)
    )
    fixed_response = retry_parser.parse_with_prompt(
        bad_response,
        parser.get_format_instructions()
    )
    return fixed_response
