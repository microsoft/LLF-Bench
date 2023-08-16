import pdb

from gpt.gpt import GPT3


gpt3 = GPT3()
response = gpt3.call_gpt(prompt="Tell me about Ludwig Van Beethoven.", max_tokens=40, temperature=0.0, logprobs=1, echo=True)
pdb.set_trace()