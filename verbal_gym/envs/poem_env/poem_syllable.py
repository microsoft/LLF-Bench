import re
from string import punctuation

import guidance
import cmudict

from verbal_gym.envs.env_wrapper import VerbalGymWrapper


def num_to_words(num):
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
                'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen',
                'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'}

    if num < 20:
        return under_20[num]
    if num < 100:
        return tens[(int)(num / 10) - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])
    # find the appropriate pivot - 'Million' in 3,603,550, or 'Thousand' in 603,550
    pivot = max([key for key in above_100.keys() if key <= num])

    return num_to_words((int)(num / pivot)) + ' ' + above_100[pivot] + (
        '' if num % pivot == 0 else ' ' + num_to_words(num % pivot))


class PoemExtractor(object):
    # use LLM to extract the poem
    # just in case more things were written
    def __init__(self, silent=True):
        self.prompt = guidance("""
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
Extract only lines of poems in the following message, ignore any part of the message that is not related to the poem.
Only return the poem line by line, including space.

```
{{content}}
```
{{~/user}}

{{#assistant~}}
{{gen 'poem' temperature=0.7}}
{{~/assistant}}
""", silent=silent)

    def __call__(self, content):
        return self.prompt(content=content)['poem']

class PoemEnv(VerbalGymWrapper):
    def __init__(self, env, syllable=7, feedback=0, silent=True):
        docstring = """
        This is a poem environment. The agent is asked to produce a poem with
        a certain number of syllables per line. The agent is given a reward"""

        super().__init__(env, docstring)
        num = num_to_words(syllable).lower()
        self.assignment = f"Can you produce a short poem where each line has {syllable} syllables?"
        self.syllable = syllable

        self.feedback = feedback
        assert self.feedback in {0, 0.5, 1}

        self.cmudict = cmudict.dict()
        self.extractor = PoemExtractor(silent=silent)

    def reset(self):
        return self.assignment

    def simple_count(self, text):
        text = text.lower()
        # remove non-alphabets
        text = re.sub('[^a-z]', '', text)

        # count syllables based on phonetics rules
        count = 0
        vowels = 'aeiouy'
        if len(text) == 0:
            return count
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le') and text[-3] not in vowels:
            count += 1
        if text.endswith('ed'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def count_syllables(self, words):
        """Use corpora to count syllables in English word or phrase."""
        # prep words for cmudict corpus
        words = words.replace('-', ' ')
        words = words.lower().split()
        num_sylls = 0
        for word in words:
            word = word.strip(punctuation)
            if word.endswith("'s") or word.endswith("’s"):
                word = word[:-2]
            result = self.cmudict[word]
            # if there is no result, we try to do a simple count
            if len(result) == 0:
                num_sylls += self.simple_count(word)
                continue

            for phonemes in self.cmudict[word][0]:
                for phoneme in phonemes:
                    if phoneme[-1].isdigit():
                        num_sylls += 1
        return num_sylls

    def get_line_feedback(self, text):
        success = True
        success_line, total_line = 0, 0
        info = []
        for i, line in enumerate(text.strip().split('\n')):
            if line == '':
                # this means it's just a segment break
                continue
            count = self.count_syllables(line)
            success *= count == self.syllable
            if count != self.syllable:
                diff = self.syllable - count  # positive: increase syllable; negative: decrease syllable
                info.append([i, line, count, diff])
            else:
                success_line += 1
            total_line += 1
        return success, success_line / total_line, info

    def step(self, a):
        # observation, reward, terminal, info
        a = self.extractor(a)
        success, frac, info = self.get_line_feedback(a)

        if success:
            return self.assignment, frac, True, {'frac': frac, 'feedback': None, 'success': 1}

        if self.feedback == 0:
            # we just say "The generated poem is not correct."
            feedback = "The generated poem is incorrect."
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where:> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = "The generated poem is incorrect.\n"
            feedback += f"This is because the poem needs to have exactly {self.syllable} syllables in each line"
            feedback += ", but lines " if len(info) > 1 else ", but line "
            for tup in info:
                i, line, count, diff = tup
                feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if len(info) > 1 else " does not."
        elif self.feedback == 1:
            # we offer a directional suggestion (you should decrease the number of syllables in this line)
            feedback = "The generated poem is incorrect.\n"
            feedback += "Here are some suggestions to fix your error:\n"
            for tup in info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += f'The line: "{line}" has {count} syllables. It should only have {self.syllable} syllables. '
                feedback += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'
        else:
            raise ValueError(f"Invalid feedback level: {self.feedback}")

        return self.assignment, frac, False, {'frac': frac, 'feedback': feedback, 'success': 1}