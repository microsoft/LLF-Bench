import openai
import time, os

API_MODE_AZURE = True

# setup openai to be either gpt3.5 or gpt4
if API_MODE_AZURE:
    openai.api_type = "azure"
    openai.api_version = "2023-05-15"
    openai.api_base = "https://nexus-openai-1.openai.azure.com/"
    openai.api_key =  os.getenv('OPEANAI_KEY')
else:
    raise NotImplementedError("OpenAI API mode not implemented")
    # openai.api_key_path =


def _call_model(messages, model, temperature, request_timeout, max_tokens=None):
    # Place one call to the model, returning the response and total number of tokens involved.
    # Minor difference between using azure service (like MSR do) or not: use `engine` or `model`

    if max_tokens is not None:
        if API_MODE_AZURE:
            response = openai.ChatCompletion.create(
                messages=messages,
                engine=model,
                temperature=temperature,
                request_timeout=request_timeout,
                max_tokens=max_tokens
            )
        else:
            response = openai.ChatCompletion.create(
                messages=messages,
                model=model,
                temperature=temperature,
                request_timeout=request_timeout,
                max_tokens=max_tokens
            )

    if API_MODE_AZURE:
        response = openai.ChatCompletion.create(
            messages=messages,
            engine=model,
            temperature=temperature,
            request_timeout=request_timeout
        )
    else:
        response = openai.ChatCompletion.create(
            messages=messages,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout
        )
    return response['choices'][0]['message']['content'], response


def call_model(messages, model, temperature, request_timeout, max_tokens=None, max_attempts=50):
    for _ in range(max_attempts):
        try:
            return _call_model(messages, model, temperature, request_timeout, max_tokens)
        except openai.error.Timeout as e:
            print(f"Request timed out: {e}")
            print("Retrying the call...")
            continue
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            # Wait the timeout period before retrying, to avoid a retry storm.
            print(f"Waiting {request_timeout} seconds before retrying...")
            time.sleep(request_timeout)
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
            #exit(1)
            #import pdb; pdb.set_trace()
            time.sleep(request_timeout)
            print("Retrying the call...")
            continue

    print_color("Failed to call the model after {} attempts.".format(max_attempts), "red")
    return None, None