from verbal_gym.utils.misc_utils import print_color
from verbal_gym.llm.gpt_models import GPT



system_prompt = """\
You are a talkative AI.\
"""
print_color('System: {}'.format(system_prompt), "red")


llm = GPT(system_prompt)

# Test Query
print('\nTest query interface.')

user_prompt= """\
Hello there! How are you?\
"""

response, info = llm.query(user_prompt)


print_color('User: {}'.format(user_prompt), "blue")
print_color('Agent: {}'.format(response))

# Test Chat
print('\nTest chat interface.')

user_prompt= """\
Hello there! I am sad. How are you?\
"""
response, info = llm.chat(user_prompt)
print_color('User: {}'.format(user_prompt), "blue")
print_color('Agent: {}'.format(response))

user_prompt= """\
What was my feeling?\
"""
response, info = llm.chat(user_prompt)
print_color('User: {}'.format(user_prompt), "blue")
print_color('Agent: {}'.format(response))
