import os
import pdb
import time
import requests
import json


class AbstractGPT:

    def __init__(self, deployment_name):

        self.api_key = os.getenv("GCR_GPT_KEY")
        self.base_url = os.getenv("GCR_GPT_URL")
        self.url = self.base_url + "/openai/deployments/" + deployment_name + "/completions?api-version=2022-12-01"

    def get_logprobs(self, prompt, logprobs=None, temperature=0.0, echo=False, MAX_WAITTIME_SEC=300):
        """
            Get logprobabilities of a given text
        """
        raise NotImplementedError()

    def call_gpt(self, prompt, max_tokens=None, logprobs=None, temperature=0.0, echo=False, MAX_WAITTIME_SEC=300):

        sleep_time = 1.0
        response = None

        while True:
            try:
                response = self._call_gpt(prompt,
                                          max_tokens=max_tokens,
                                          logprobs=logprobs,
                                          temperature=temperature,
                                          echo=echo)
            except:
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
        formatted_response = json.dumps(response, indent=4)

        return formatted_response["choices"][0]
