import re

class SimpleGuidanceParser:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, template_text, **kwargs):
        labeled_blocks = self.extract_blocks(template_text)
        if labeled_blocks[-1][0] == "assistant":
            labeled_blocks = labeled_blocks[:-1] # we remove the last assistant block, because that's for generation

        content = template_text
        for block_type, content in labeled_blocks:
            # if statement is handled first, because this decides if the content should stay or disappear
            content = self.parse_if_block(content, **kwargs)
            content = self.populate_template_for_each(content, **kwargs)
            content = self.populate_vars(content, **kwargs)
            if self.verbose:
                print("------New block------")
                print(f"Block Type: {block_type}")
                print(content)
                print("------End block------")

        return content

    def parse_if_block(self, parsed_text, **kwargs):
        # Regular expression to capture the content inside the {{#if ...}} and {{/if}} tags
        pattern = r"{{#if (\w+)}}(.*?){{/if}}"

        matches = re.findall(pattern, parsed_text, re.DOTALL)
        for condition_var, block_content in matches:
            # If the condition variable exists in kwargs and its value is True
            if condition_var not in kwargs:
                raise Exception(f"Key '{condition_var}' for if statement not found in provided arguments.")
            cond = kwargs[condition_var]
            if cond:
                # Remove the {{#if}} and {{/if}} tags, but keep the content
                parsed_text = parsed_text.replace(r"{{#if " + condition_var + "}}" + block_content + "{{/if}}",
                                                  block_content.strip())
            else:
                # Remove the entire block
                parsed_text = parsed_text.replace(r"{{#if " + condition_var + "}}" + block_content + "{{/if}}", "")

        # Return the modified text
        return parsed_text

    def populate_vars(self, template, **kwargs):
        # Regular expression to find all placeholders
        placeholders = re.findall(r"{{(.*?)}}", template)

        # Replace each placeholder with its corresponding value from kwargs
        for placeholder in placeholders:
            if placeholder in kwargs:
                template = template.replace(f"{{{{{placeholder}}}}}", kwargs[placeholder])
            else:
                template = template.replace(f"{{{{{placeholder}}}}}", f"Placeholder {placeholder} not provided")
                raise Exception(template)

        return template

    def populate_template_for_each(self, template, **kwargs):
        # We don't support nested for-loop

        each_key_match = re.search(r"~#each (\w+)", template)
        if not each_key_match:
            return template

        each_key = each_key_match.group(1)

        if each_key not in kwargs:
            raise Exception(f"Key '{each_key}' for each statement not found in provided arguments.")

        examples = kwargs[each_key]

        # Regular expression to extract keys after 'this.'
        keys = re.findall(r"{{this\.(.*?)}}", template)

        # Getting the template part inside the {{~#each}} and {{~/each}} tags
        template_inside_each = re.search(r"{{~#each \w+}}(.*?){{~/each}}", template, re.DOTALL).group(1).strip()

        # Generating the text for each dictionary in examples
        populated_texts = []
        for example in examples:
            populated_text = template_inside_each
            for key in keys:
                if key in example:
                    populated_text = populated_text.replace("{{this."+key+"}}", example[key])
            populated_texts.append(populated_text)

        return "\n".join(populated_texts)

    def extract_blocks(self, parsed_text):
        # Define regex patterns for each block type
        patterns = {
            "system": re.compile(r"{{#system~}}(.*?){{~/system}}", re.DOTALL),
            "user": re.compile(r"{{#user~}}(.*?){{~/user}}", re.DOTALL),
            "assistant": re.compile(r"{{#assistant~}}(.*?){{~/assistant}}", re.DOTALL)
        }

        # Find all occurrences of each block and label them
        labeled_blocks = []
        for block_type, pattern in patterns.items():
            for match in pattern.findall(parsed_text):
                labeled_blocks.append((block_type, match.strip()))

        # Sort by their appearance order in the parsed text
        labeled_blocks.sort(key=lambda x: parsed_text.index(x[1]))

        return labeled_blocks

def usage_test_1():
    # Test
    parsed_text = """
    {{#user~}}

    Now, you are given a new assignment, and you want to see if you can update the instructions to help the student write a poem that satisfies the new assignment.

    {{#if exists_instruction}}
    In addition, here are some helpful advice and guidance:
    {{instruction}}
    {{/if}}

    Your Instruction:
    {{~/user}}
    """

    kwargs = {
        "exists_instruction": True,
        "instruction": "Try to use metaphors and similes to add depth to your poem."
    }

    parser = SimpleGuidanceParser()
    results = parser(parsed_text, **kwargs)

    print(results)

def usage_test_2():
    parsed_text = """
    {{#system~}}
    You are a helpful assistant that wants to come up with instructions to a student to help them write a poem that is satisfactory to a teacher's assignment.
    The student's poem needs to satisfy the requirement of this assignment.
    {{~/system}}

    {{#user~}}
    Here are some instructions you wrote for the previous assignments:
    {{~#each examples}}
    Teacher's Assignment: {{this.assignment}}

    Your Instruction: 
    {{this.instruction}}
    ---------------
    {{~/each}}
    {{~/user}}

    {{#user~}}

    Now, you are given a new assignment, and you want to see if you can update the instructions to help the student write a poem that satisfies the new assignment.
    Teacher's Assignment: {{new_assignment}}

    Your Instruction:
    {{~/user}}
    """
    examples = [
        {"assignment": "Write about a rainy day.", "instruction": "Imagine the sound of raindrops..."},
        {"assignment": "Describe a sunny day.", "instruction": "Think of the warmth of the sun..."}
    ]

    new_assignment = "Compose a poem about winter."
    parser = SimpleGuidanceParser()
    results = parser(parsed_text, examples=examples, new_assignment=new_assignment)
    print(results)

if __name__ == '__main__':
    usage_test_1()
    usage_test_2()