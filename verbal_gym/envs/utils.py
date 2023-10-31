import numpy as np

from typing import Dict, Any, Tuple, Union, List, Callable
from verbal_gym.llm import make_llm

def format(prompts : List[str], method=Union[str, int, Callable[[List[str],  Dict[str,str]], str]], **kwargs : Dict[str,str]):
    """ A helper method for selecting from a set of paraphrased prompts.

        Args:
            prompts: A list of prompt templates to select from. The first one is
            the default.

            method: The method to use in selecting the prompt. It can be either
            'random', 'llm', or an integer. If it is 'random', a template would
            be randomly selected from `prompts`. If it is 'llm', DEFAULT_LLM
            would be used to paraphrased the first (i.e. default) template in
            `prompts`. If it is an integer, it is used as the index to select
            from the template in `prompts`.

            **kwargs: The keyword arguments to be used in formatting the template.

    """

    if method=='random':
        return np.random.choice(prompts).format(**kwargs)
    elif method=='llm':
        DEFAULT_MODEL = "gcr:gpt-35-turbo"
        DEFAULT_LLM = make_llm(DEFAULT_MODEL, system_prompt="You're a helpful assistant.", temperature=0.0)
        template, _ = DEFAULT_LLM.generate(
            f" Paraphrase the following sentence, while keeping the pythin syntax.\n{prompts[0]}"
        )
        return template.format(**kwargs)
    else:
        assert type(method)==int, "The method must be either 'random', 'llm', a callable, or an integer."
        idx = method
        return prompts[idx % len(prompts)].format(**kwargs)
