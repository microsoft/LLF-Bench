# Poem Environment

Each poem environment has an attribute called "assignment" that is the barebone description of what the poem should be.
It usually only contains the name of the poetry form it wants the model to generate.

We introduce two poem environments:
1. `LineSyllableConstrainedPoem` with two specific instantiations: Haiku (5-7-5) and Tanka (5-7-5-7-5), but we can specify any kind of line number + syllable constraints.
2. `SyllableConstrainedPoem`: requires each generated line to have the same number of syllables, but not requirement on number of lines.

In order to use these environments, we need to install the following packages:
1. `pip install cmudict`
2. `pip install guidance`
3. `pip install syllables`

Here are some examples of how to use these environments:

```python
poem = """
Beneath autumn trees,
Crimson leaves dance on the wind,
Whispers of goodbye.
As nature's gown turns to gold,
I'm left longing for the spring.
"""

poem_env = Tanka(feedback=1)

poem_env.step(poem)

('Can you write me a Tanka? A Tanka is a poem that consists of five lines. The number of syllables per line is in a pattern of 5-7-5-7-7.',
 1.0,
 True,
 {'feedback': ''})

poem_env = SyllableConstrainedPoem(feedback=1)
poem_env.step(poem)

('Can you produce a short poem where each line has 7 syllables?',
 0.6,
 False,
 {'frac': 0.6,
  'feedback': 'The generated poem is incorrect.\nHere are some suggestions to fix your error:\nThe line: "Beneath autumn trees," has 5 syllables. It should only have 7 syllables. You should rewrite the line to have more syllables.\nThe line: "Whispers of goodbye." has 5 syllables. It should only have 7 syllables. You should rewrite the line to have more syllables.\n',
  'success': 1})
```