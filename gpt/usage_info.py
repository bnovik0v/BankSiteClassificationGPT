""" Decorator to add usage info to the response """

from typing import Tuple, Dict, Any

from langchain.callbacks import get_openai_callback


def return_usage_info(func):
    """ Add usage info to the response """

    def wrapper(*args, **kwargs) -> Tuple[Any, Dict]:
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)

        usage_info = {
            'total_cost': cb.total_cost,
            'total_tokens': cb.total_tokens
        }

        return result, usage_info

    return wrapper