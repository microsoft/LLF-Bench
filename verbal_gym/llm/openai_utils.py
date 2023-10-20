import openai
import time, os
from verbal_gym.utils.misc_utils import print_color

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
            openai.api_base = "https://nexus-openai-1.openai.azure.com/"
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