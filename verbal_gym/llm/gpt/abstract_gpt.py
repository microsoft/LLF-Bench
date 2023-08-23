import os
import pdb
import time
import requests
import json
from verbal_gym.llm.llm import LLM


class AbstractGPT(LLM):

    def __init__(self, deployment_name, system_prompt=''):
        super().__init__(system_prompt=system_prompt)

        self.api_key = os.getenv("GCR_GPT_KEY")
        self.base_url = os.getenv("GCR_GPT_URL")
        self.url = self.base_url + "/openai/deployments/" + deployment_name + "/completions?api-version=2022-12-01"

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        pass  # Since it does not have a history.

    def logprob(self, prompt, *args, **kwargs):
        return self.get_logprobs(prompt, *args, **kwargs)

    def get_logprobs(self, prompt, MAX_WAITTIME_SEC=300):
        """
            Get log-probabilities of the given prompt
        """
        response = self.call_gpt(prompt,
                                 max_tokens=0,
                                 logprobs=1,
                                 temperature=0.0,
                                 echo=True,
                                 MAX_WAITTIME_SEC=MAX_WAITTIME_SEC)

        # GPT does not compute logprob of the first token
        first_tk_logprob = response["choices"][0]["logprobs"]["token_logprobs"][0]
        assert first_tk_logprob == 0 or first_tk_logprob is None
        logprob = sum(response["choices"][0]["logprobs"]["token_logprobs"][1:])

        return logprob

    def generate(self, prompt, max_tokens=None, logprobs=None, temperature=0.0, echo=False, MAX_WAITTIME_SEC=300):
        """
            Call GPT in a retrying mode

            :param max_tokens: Maximum number of tokens to generate
            :param logprobs: None if dont want logprobs to be returned, otherwise,
                             an integer k to return the top-k logprobs
            :param echo: A boolean which if True returns logprobs also of the prompt, otherwise,
                        return logprobs (if not None) only for the generation
            :param MAX_WAITTIME_SEC: Maximum time to wait if failure occurs before re-trying

            returns: text, info

            where text is the generation and info is a dictionary where info["total_logprobs"] is the total logprobs of
            the entire text if echo is True, otherwise, only of the generation.
        """

        response = self.call_gpt(prompt,
                                 max_tokens=max_tokens,
                                 logprobs=logprobs,
                                 temperature=temperature,
                                 echo=echo,
                                 MAX_WAITTIME_SEC=MAX_WAITTIME_SEC)

        text = response["choices"][0]["text"]

        if logprobs is not None:
            if echo:
                # GPT does not compute logprob of the first token
                first_tk_logprob = response["choices"][0]["logprobs"]["token_logprobs"][0]
                assert first_tk_logprob == 0 or first_tk_logprob is None
                logprob = sum(response["choices"][0]["logprobs"]["token_logprobs"][1:])
            else:
                logprob = sum(response["choices"][0]["logprobs"]["token_logprobs"])
        else:
            logprob = None

        info = {
            "logprob": logprob,
            "echo": echo
        }

        return text, info

    def call_gpt(self, prompt, max_tokens=None, logprobs=None, temperature=0.0, echo=False, MAX_WAITTIME_SEC=300):
        """
            Call GPT in a retrying mode

            :param max_tokens: Maximum number of tokens to generate
            :param logprobs: None if dont want logprobs to be returned, otherwise,
                             an integer k to return the top-k logprobs
            :param echo: A boolean which if True returns logprobs also of the prompt, otherwise,
                        return logprobs (if not None) only for the generation
            :param MAX_WAITTIME_SEC: Maximum time to wait if failure occurs before re-trying

            returns: response

            where response is a dictionary returned by GPT call.
        """

        sleep_time = 1.0
        response = None

        while True:
            try:
                response = self._call_gpt(prompt,
                                          max_tokens=max_tokens,
                                          logprobs=logprobs,
                                          temperature=temperature,
                                          echo=echo)
                break
            except Exception as e:
                print("Error ", e)
                sleep_time = min(sleep_time, MAX_WAITTIME_SEC)
                time.sleep(sleep_time)
                sleep_time = 2 * sleep_time
                continue

        return response

    def _call_gpt(self, prompt, max_tokens=None, logprobs=None, temperature=0.0, echo=False):

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "echo": echo
        }

        if logprobs is not None:
            assert type(logprobs) == int
            payload["logprobs"] = logprobs

        if max_tokens is not None:
            assert type(max_tokens) == int
            payload["max_tokens"] = max_tokens

        r = requests.post(self.url,
                          headers={
                              "api-key": self.api_key,
                              "Content-Type": "application/json"
                          },
                          json=payload
                          )

        response = json.loads(r.text)

        return response
