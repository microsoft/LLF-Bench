import re
from string import punctuation

import gym
import cmudict
import guidance
import syllables
import sys

# gotta add rhyme to it

class PoemUtil:
    # designed as a Mixin class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.cmudict = cmudict.dict()

    def simple_syllable_count(self, word):
        # can also use pip syllables library
        text = word.lower()
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

    def count_syllables(self, line):
        """Use corpora to count syllables in English word or phrase."""
        # prep words for cmudict corpus
        line = line.replace('-', ' ')
        words = line.lower().split()
        num_sylls = 0
        for word in words:
            word = word.strip(punctuation)
            if word.endswith("'s") or word.endswith("’s"):
                word = word[:-2]
            # if word in missing_words:
            #     num_sylls += missing_words[word]
            result = self.cmudict[word]
            # if there is no result, we try to do a simple count
            if len(result) == 0:
                # heuristic based checking
                num_sylls += syllables.estimate(word)  # simple_syllable_count(word)
                continue

            for phonemes in self.cmudict[word][0]:
                for phoneme in phonemes:
                    if phoneme[-1].isdigit():
                        num_sylls += 1
        return num_sylls

    def seed(self, seed):
        pass

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

class Haiku(PoemUtil, gym.Env):
    def __init__(self, feedback=0, silent=True, use_extractor=False):
        self.assignment = f"Can you write me a haiku? A haiku is a poem that consists of three phrases composed of 17 syllables in a 5, 7, 5 pattern."
        self.form_name = 'Haiku'
        self.use_extractor = use_extractor
        self.extractor = None

        self.feedback = feedback
        self.syllable_req = [5, 7, 5]
        self.syllable_req_str = [str(i) for i in self.syllable_req]
        assert feedback in {0, 0.5, 1}

        self.action_space = gym.spaces.Text(sys.maxsize)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        super().__init__()

    def reset(self, **kwargs):
        return self.assignment

    def initialize_text_extractor(self, poem_extractor: PoemExtractor):
        self.extractor = poem_extractor

    def line_number_incorrect(self, observed_num):
        if self.feedback == 0:
            return f"The generated {self.form_name} is incorrect."
        elif self.feedback == 0.5:
            return f"The generated {self.form_name} is incorrect. This is because the {self.form_name} needs to have exactly {len(self.syllable_req)} lines."
        elif self.feedback == 1:
            improv_direction = "more" if observed_num < len(self.syllable_req) else "less"
            return f"The generated {self.form_name} is incorrect. This is because the {self.form_name} needs to have exactly {len(self.syllable_req)} lines. You wrote {observed_num} lines. Write {improv_direction} lines."
        else:
            raise ValueError(f"Invalid feedback level: {self.feedback}")

    def line_syllable_check(self, lines):
        success = True
        success_line, total_line = 0, 0
        info = []

        for i in range(len(self.syllable_req)):
            # this is to say -- if the generated poem is shorter than required lines
            # we just count the missing line as wrong (for the missing line)
            if i >= len(lines):
                sucess = False
                total_line += 1
                continue

            line = lines[i]
            count = self.count_syllables(line)
            success *= count == self.syllable_req[i]
            if count != self.syllable_req[i]:
                diff = self.syllable_req[i] - count  # positive: increase syllable; negative: decrease syllable
                info.append([i, line, count, diff])
            else:
                success_line += 1
            total_line += 1

        return success, success_line / total_line, info

    def produce_line_feedback(self, info):
        if self.feedback == 0:
            # we just say "The generated poem is not correct."
            feedback = f"The generated {self.form_name} is incorrect."
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where:> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = f"The generated {self.form_name} is incorrect.\n"
            feedback += f"This is because {self.form_name} needs to have exactly {'-'.join(self.syllable_req_str)} syllables in three lines"
            feedback += ", but lines " if len(info) > 1 else ", but line "
            for tup in info:
                i, line, count, diff = tup
                # feedback +=  f'The line: "{line}" has {count} syllables. It should only have {self.syllable} syllables' + '\n'
                feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if len(info) > 1 else " does not."
        elif self.feedback == 1:
            # we offer a directional suggestion (you should decrease the number of syllables in this line)
            feedback = f"The generated {self.form_name} is incorrect.\n"
            feedback += "Here are some suggestions to fix your error:\n"
            for tup in info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += f'The line: "{line}" has {count} syllables. It should only have {self.syllable_req[line]} syllables. '
                feedback += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'
        return feedback

    def step(self, a):
        # observation, reward, terminal, info

        if self.use_extractor:
            if self.extractor is None:
                raise Exception(
                    "Must pass in an extractor through initialize_text_extractor before using the extractor.")
            a = self.extractor(a)

        feedbacks = []
        success = True

        lines = []
        for line in a.strip().split('\n'):
            if line == '':
                continue
            lines.append(line)

        if len(lines) != len(self.syllable_req):
            success = False
            feedbacks.append(self.line_number_incorrect(len(lines)))

        syllabel_success, frac, info = self.line_syllable_check(lines)
        success *= syllabel_success

        if len(info) > 0:
            feedbacks.append(self.produce_line_feedback(info))

        if len(feedbacks) == 0:
            feedback = "Congrats! You have successfully produced a poem that matches the assignment description."
        elif self.feedback == 0:
            feedback = feedbacks[-1]
        else:
            feedback = "\n".join(feedbacks)

        terminal = False   # one step environment

        return self.assignment, frac, terminal, {'feedback': feedback, 'success': int(success)}


class Tanka(Haiku):
    def __init__(self, feedback=0, silent=True, use_extractor=False):
        # We can extend this to add "theme" of the poem
        # This increases difficulty a little, but also hard to check if it's thematic or not.
        super().__init__(feedback, silent, use_extractor)
        self.assignment = f"Can you write me a Tanka? A Tanka is a poem that consists of five lines composed of syllables in a 5-7-5-7-7 pattern."
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_req = [5, 7, 5, 7, 7]
        self.syllable_req_str = [str(i) for i in self.syllable_req]
        self.form_name = 'Tanka'


class LineSyllableConstrainedPoem(Haiku):
    def __init__(self, syllable_req=[7, 7, 7], feedback=0, silent=True, use_extractor=False):
        # We can extend this to add "theme" of the poem
        # This increases difficulty a little, but also hard to check if it's thematic or not.
        super().__init__(feedback, silent, use_extractor)
        self.syllable_req_str = [str(i) for i in syllable_req]
        self.assignment = f"Can you write me a poem? It should have {len(syllable_req)} lines. The number of syllables for the lines in the poem should follow a {'-'.join(self.syllable_req_str)} pattern."
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_req = syllable_req
        self.form_name = 'poem'

class SyllableConstrainedPoem(PoemUtil, gym.Env):
    def __init__(self, syllable=7, feedback=0, silent=True, use_extractor=False):

        super().__init__()
        self.assignment = f"Can you produce a short poem where each line has {syllable} syllables?"
        self.syllable = syllable
        self.use_extractor = use_extractor

        self.feedback = feedback
        assert self.feedback in {0, 0.5, 1}

        self.cmudict = cmudict.dict()
        self.extractor = None

        self.action_space = gym.spaces.Text(sys.maxsize)
        self.observation_space = gym.spaces.Text(sys.maxsize)

    def reset(self, **kwargs):
        return self.assignment

    def initialize_text_extractor(self, poem_extractor: PoemExtractor):
        self.extractor = poem_extractor

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
        if self.use_extractor:
            if self.extractor is None:
                raise Exception("Must pass in an extractor through initialize_text_extractor before using the extractor.")
            a = self.extractor(a)
        success, frac, info = self.get_line_feedback(a)

        if success:
            feedback = "Congrats! You have successfully produced a poem that matches the assignment description."
            return self.assignment, frac, False, {'frac': frac, 'feedback': feedback, 'success': 1}

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

        terminal = False   # one step environment

        out =  self.assignment, frac, terminal, {'frac': frac, 'feedback': feedback, 'success': 0}
        return out