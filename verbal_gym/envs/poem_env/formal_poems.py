import re
from string import punctuation

import gym
import cmudict
import syllables
import sys

from verbal_gym.agents.parser_util import SimpleGuidanceParser
from verbal_gym.envs.poem_env.prompts import *
from verbal_gym.envs.utils import format

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
    def __init__(self, llm, silent=True):
        self.llm = llm
        self.prompt = SimpleGuidanceParser("""
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
""")

    def __call__(self, content):
        messages = self.prompt(content=content)
        response, info = self.llm.generate(messages)
        return response

class Haiku(PoemUtil, gym.Env):
    def __init__(self, feedback=0, silent=True, use_extractor=False):
        self.assignment = format(haiku_b_instruction)
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
            return format(r_feedback_neg, form_name=self.form_name)
        elif self.feedback == 0.5:
            return format(hn_feedback, form_name=self.form_name, k=len(self.syllable_req))
        elif self.feedback == 1:  # NOTE though marked as fp, this is hn+fp.
            improv_direction = "more" if observed_num < len(self.syllable_req) else "less"
            return format(fp_feedback, form_name=self.form_name, k=len(self.syllable_req), observed_num=observed_num,
                          improv_direction=improv_direction)
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
            feedback = format(r_feedback_neg, form_name=self.form_name) + '\n'
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where:> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = format(r_feedback_neg, form_name=self.form_name)+ '\n'
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
            feedback = format(r_feedback_neg, form_name=self.form_name)+ '\n'
            feedback += format(line_fp_feedback_1) + '\n'
            for tup in info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += format(line_fp_feedback_2, line=line, count=count,
                                    k=self.syllable_req[i], improv_direction=improv_direction) + '\n'
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
            feedback = format(r_feedback_pos)
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
        self.assignment = format(tanka_b_instruction)
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
        self.assignment = format(line_syb_constrained_poem_b_instruction, k=len(syllable_req), pattern='-'.join(self.syllable_req_str))
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_req = syllable_req
        self.form_name = 'poem'

class SyllableConstrainedPoem(PoemUtil, gym.Env):
    def __init__(self, syllable=7, feedback=0, silent=True, use_extractor=False):

        super().__init__()
        self.assignment =  format(syllable_constrained_b_instruction, syllable=syllable)
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
            feedback = format(r_feedback_pos)
            return self.assignment, frac, False, {'frac': frac, 'feedback': feedback, 'success': 1}

        if self.feedback == 0:
            # we just say "The generated poem is not correct."
            feedback = format(r_feedback_neg, form_name="poem")
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where:> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = format(r_feedback_neg, form_name="poem") + '\n'
            feedback += f"This is because the poem needs to have exactly {self.syllable} syllables in each line"
            feedback += ", but lines " if len(info) > 1 else ", but line "
            for tup in info:
                i, line, count, diff = tup
                feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if len(info) > 1 else " does not."
        elif self.feedback == 1:
            # we offer a directional suggestion (you should decrease the number of syllables in this line)
            feedback = format(r_feedback_neg, form_name="poem") + '\n'
            feedback += format(line_fp_feedback_1) + '\n'
            for tup in info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += format(line_fp_feedback_2, line=line, count=count,
                                    k=self.syllable_req[i], improv_direction=improv_direction) + '\n'
        else:
            raise ValueError(f"Invalid feedback level: {self.feedback}")

        terminal = False   # one step environment

        out =  self.assignment, frac, terminal, {'frac': frac, 'feedback': feedback, 'success': 0}
        return out