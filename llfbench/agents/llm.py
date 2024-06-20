from abc import ABC
import openai
import time, os
from llfbench.agents.utils import print_color

class LLM(ABC):
    """ This class represents a black box LLM. """
    # NOTE when subclassing this class, please include **kwargs in each method.

    def __init__(self, system_prompt='', **kwargs):
        """ Initialize the LLM """
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        raise NotImplementedError

    def chat(self,
             prompt, *,
             logprob=None, # None or int. If int, return top-k logprobs.
             timeout,  # Maximum time to wait if failure occurs before re-trying
             temperature,  # temperature of the generation
             max_tokens,  # maximum number of tokens to generate
             max_attempts):  # maximum number of attempts to call the model
        """ This is a history-dependent chat response. """
        raise NotImplementedError
        # info is dict with keys 'logprobs' and 'response', where 'response' is the original response object from the model.
        return generation, info

    def generate(self,
                 prompt, *,
                 logprob=None, # None or int. If int, return top-k logprobs.
                 timeout,  # Maximum time to wait if failure occurs before re-trying
                 temperature,  # temperature of the generation
                 max_tokens,  # maximum number of tokens to generate
                 max_attempts):  # maximum number of attempts to call the model

        """ This is one-time query response. """
        raise NotImplementedError
        # info is dict with keys 'logprobs' and 'response', where 'response' is the original response object from the model.        return response, info
        return generation, info


    def logprob(self,
                prompt, *,
                timeout,  # Maximum time to wait if failure occurs before re-trying
                temperature,  # temperature of the generation
                max_tokens,  # maximum number of tokens to generate
                max_attempts):  # maximum number of attempts to call the model
        """ This is one-time query response. """
        raise NotImplementedError
        return logprob


OPENAI_API_INITIALIZED = False
API_MODE_AZURE = True

def _call_model(messages, model, temperature, timeout, logprobs=None, max_tokens=None):

    backend, model = model.split(':')
    assert backend in ('azure', 'openai')
    init_openai_api(api_mode_azure=(backend=='azure'))

    # Place one call to the model, returning the response and total number of tokens involved.
    # Minor difference between using azure service (like MSR do) or not: use `engine` or `model`
    config = dict(
        messages=messages,
        temperature=temperature,
        request_timeout=timeout,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    if max_tokens is not None:
        config['max_tokens'] = max_tokens
    if API_MODE_AZURE:
        model = model.replace('3.5', '35')
        config['engine'] = model
    else:
        model = model.replace('35', '3.5')
        config['model'] = model

    if model in ('text-davinci-003'):  # legacy models
        prompt = ''
        for m in config['messages']:
            if len(m['content']) > 0:
                prompt += m['role'] + ': ' + m['content'] + '\n'
        config['prompt'] = prompt
        del config['messages']
        response = openai.Completion.create(**config)
        logprobs = response["choices"][0]["logprobs"]
        info = {'logprobs': logprobs, 'response': response}
        return response['choices'][0]['text'], info
    else:
        del config['logprobs']  # logprobs is not supported by GPT3.5 or newer
        response = openai.ChatCompletion.create(**config)
        info = {'logprobs': None, 'response': response}
        return response['choices'][0]['message'].get('content', ''), info


def init_openai_api(api_mode_azure=True):
    global OPENAI_API_INITIALIZED, API_MODE_AZURE
    if not OPENAI_API_INITIALIZED:
        OPENAI_API_INITIALIZED = True
        API_MODE_AZURE = api_mode_azure
        # setup openai to be either gpt3.5 or gpt4
        if API_MODE_AZURE:
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            openai.api_base = os.getenv('AZURE_OPENAI_BASE')
            openai.api_key = os.getenv('AZURE_OPENAI_KEY')
        else:
            openai.api_key_path = os.getenv('OPENAI_KEY_PATH')


def call_model(messages, model, temperature, timeout, wait_time=2, max_tokens=None, logprobs=None, max_attempts=float('inf')):
    i = 0
    while i < max_attempts:
        i+=1
        try:
            return _call_model(messages, model, temperature, timeout, max_tokens)
        except openai.error.Timeout as e:
            print(f"Request timed out: {e}")
            print("Retrying the call...")
            continue
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            # Wait the timeout period before retrying, to avoid a retry storm.
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            print("Retrying the call...")
            continue
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            exit(1)
        except openai.error.AuthenticationError as e:
            print(f"OpenAI API returned an Authentication Error: {e}")
            exit(1)
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            exit(1)
        except openai.error.InvalidRequestError as e:
            print(f"Invalid Request Error: {e}")
            exit(1)
        except openai.error.ServiceUnavailableError as e:
            print(f"Service Unavailable: {e}")
            exit(1)
        except Exception as e:
            print(f"Unexpected exception: {e}")
            print(f"Message: {messages}")
            #exit(1)
            #import pdb; pdb.set_trace()
            time.sleep(timeout)
            print("Retrying the call...")
            continue

    print_color("Failed to call the model after {} attempts.".format(max_attempts), "red")
    return None, None

class GPT(LLM):
    """ This is based on ChatCompletion api. """

    def __init__(self,
                 system_prompt='', *,
                 model="gpt-35-turbo",
                 temperature=0.0,
                 timeout=100,
                 max_tokens=None,
                 max_attempts=50,
                 **kwargs):
        """ Initialize the LLM """
        super().__init__(system_prompt)
        self.spec = dict(
            model = model,
            temperature = temperature,
            timeout = timeout,
            max_tokens = max_tokens,
            max_attempts = max_attempts,
        )

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, prompt, **kwargs):
        """ This is a history-dependent response. """
        self.messages.append({"role": "user", "content": prompt})
        spec = self.spec.copy()
        spec.update(kwargs)
        return call_model(self.messages, **spec)

    def generate(self, prompt, **kwargs):
        """ This is one-time query response. """
        if isinstance(prompt, str):
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            assert type(prompt) == list, "we also accept a list of messages or a single message, but this is not"
            messages = prompt
        spec = self.spec.copy()
        spec.update(kwargs)
        return call_model(messages, **spec)



import autogen
class Autgen(LLM):
    """ This is based on ChatCompletion api. """

    def __init__(self,
                 system_prompt='', *,
                 model="gpt-35-turbo",
                 config_list=None,
                 **kwargs):
        """ Initialize the LLM """
        super().__init__(system_prompt)

        if config_list is None:
            config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
            model = model.replace('35', '3.5')
            config_list = [config for config in config_list if config['model'] == model]
            assert len(config_list) == 1, f"Model {model} not found in the config list."
        self.llm = autogen.OpenAIWrapper(config_list=config_list)

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        pass

    def chat(self, prompt, **kwargs):
        """ This is a history-dependent response. """
        raise NotImplementedError

    def generate(self, prompt, **kwargs):
        """ This is one-time query response. """
        return self.call_llm(prompt, verbose=False)


    def call_llm(self, prompt, verbose=False):  # TODO Get this from utils?
        """Call the LLM with a prompt and return the response."""
        if verbose:
            print("Prompt\n", prompt)

        # try:  # Try tp force it to be a json object
        #     response = self.llm.create(
        #         messages=[
        #             {
        #                 "role": "system",
        #                 "content": self.system_prompt,
        #             },
        #             {
        #                 "role": "user",
        #                 "content": prompt,
        #             }
        #         ],
        #         response_format={"type": "json_object"},
        #     )
        # except Exception:
        response = self.llm.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        response = response.choices[0].message.content

        if verbose:
            print("LLM response:\n", response)
        return response, None


def standardize_model_name(model):
    if model in ('gpt-35', 'gpt-3.5', 'gpt-35-turbo', 'gpt-3.5-turbo'):
        model = 'gpt-35-turbo'
    if model in ('gpt-3'):
        model = 'text-davinci-003'
    return model


def make_llm(model, **kwargs):
    """ model = backend:model_name """
    available_backends = ['gcr']  # TODO
    if os.getenv('AZURE_OPENAI_KEY') is not None:
        available_backends.append('azure')

    if os.getenv('OPENAI_KEY_PATH') is not None:
        available_backends.append('openai')

    if autogen.config_list_from_json("OAI_CONFIG_LIST") is not None:
        available_backends.append('autogen')

    backend = None
    if':' in model:  # azure; autogen
        backend, model = model.split(':')
    else:
        backend = 'openai'

    if len(available_backends)==0:
        raise Warning("No LLM backend is set up. Please set up at least one backend.")

    if backend not in available_backends:
        # Try to use the first available backend if the requested one is not available
        Warning(f"{backend} backend is not set up. Switch to {available_backends[0]} backend.")
        backend = available_backends[0]

    model = standardize_model_name(model)

    if backend in ('azure', 'openai'):
        return GPT(model=backend+':'+model, **kwargs)
    elif backend == 'autogen':
        return Autgen(model=model, **kwargs)
    else:
        raise ValueError("Unknown LLM model: {}".format(model))

# Create a default LLM
DEFAULT_MODEL = "gpt-35-turbo"
DEFAULT_LLM = make_llm(DEFAULT_MODEL, system_prompt="You're a helpful assistant.", temperature=0.0)