
# This file contains the prompts for the verbal instructions and feedback.

loss_b_instruction_template = "You are trying to minimize the output (y) of a function by choosing input (x). The goal is to choose x such that y is as small as possible."
loss_b_instruction = (
    "Your objective is to reduce the function's output (y) by selecting an appropriate input (x), aiming to make y as minimal as possible.",
    "You aim to find an input (x) that minimizes the output (y) of a function, striving to achieve the smallest possible value for y.",
    "The task is to identify an input (x) that results in the lowest possible output (y) from a function, minimizing y to the greatest extent possible.",
    "You are working to choose an input (x) that leads to the smallest possible output (y) from a function, with the ultimate goal of minimizing y.",
    "Your goal is to select an input (x) that will minimize the function’s output (y), ensuring that y is as low as it can be.",
    "You are in the process of determining the best input (x) to minimize the output (y) of a function, aiming for the smallest y value achievable.",
    "The objective is to find the optimal input (x) that will yield the smallest possible output (y) from a function, in an effort to minimize y.",
    "You're on a quest to choose an input (x) that reduces the function's output (y) to its lowest possible level, minimizing y.",
    "Your mission is to select an input (x) that brings the output (y) of a function to its minimum, ensuring y is as small as it can be.",
    "You are working towards identifying an input (x) that minimizes the output (y) of a function, with the ultimate aim of achieving the smallest y possible."
)

r_feedback_pos_template = "You have reached the minimum!"
r_feedback_pos = (
    "You've arrived at the lowest point!",
    "The minimum has been achieved!",
    "You have attained the smallest possible value!",
    "The lowest level has been reached!",
    "You’ve hit rock bottom!",
    "You’ve successfully reached the minimum value!",
    "The minimal point has been attained!",
    "You have successfully achieved the lowest possible level!",
    "The minimum threshold has been reached!",
    "You’ve reached the lowest possible point!"
)

r_feedback_neg_template = "You have not reached the minimum!"
r_feedback_neg = (
    "You haven't attained the lowest point yet!",
    "The minimum has yet to be achieved!",
    "You are still above the smallest possible value!",
    "The lowest level hasn’t been reached!",
    "You’ve not hit the minimal point!",
    "The minimum value remains elusive!",
    "You haven’t managed to reach the lowest level yet!",
    "The minimal threshold is still ahead!",
    "You are yet to achieve the minimum!",
    "The lowest possible point hasn’t been reached!"
)

hp_feedback_dim1_template = "You chose {action} from {prev_x}. {increasing} the first number {prev_x} is correct."
hp_feedback_dim1 = (
    "Your selection was {action}, starting from {prev_x}. It's right to {increasing} the initial number {prev_x}.",
    "Opting for {action} from {prev_x} was your decision. Indeed, {increasing} the opening number {prev_x} is accurate.",
    "You decided on {action} from {prev_x}. Correctly, you are {increasing} the first numeral {prev_x}.",
    "Your choice was {action}, taking off from {prev_x}. Aptly, you’ve been {increasing} the introductory number {prev_x}.",
    "{action} was your pick from {prev_x}. It's accurate to say that {increasing} the initial digit {prev_x} is proper.",
    "From {prev_x}, you selected {action}. It is indeed correct to {increasing} the first figure {prev_x}.",
    "You opted for {action} starting at {prev_x}. It’s right on target to {increasing} the primary number {prev_x}.",
    "You went with {action} from {prev_x}. It is correct to {increasing} the starting number {prev_x}.",
    "Choosing {action} from {prev_x} was your move. You are right in {increasing} the first number {prev_x}.",
    "From the starting point of {prev_x}, you picked {action}. Correctly, you have been {increasing} the first digit {prev_x}."
)

hp_feedback_dim2_template = "You chose {action} from {prev_x}. {increasing} the second number {prev_x} is correct."
hp_feedback_dim2 = (
    "Your selection was {action} starting from {prev_x}. It's right to {increasing} the second figure {prev_x}.",
    "Opting for {action} from {prev_x} was your choice. Correctly, you’ve been {increasing} the second digit {prev_x}.",
    "You decided on {action} from the point of {prev_x}. Aptly, {increasing} the second value {prev_x} is accurate.",
    "The choice of {action} from {prev_x} was made by you. Indeed, {increasing} the second number {prev_x} is the right move.",
    "Selecting {action} from {prev_x} was your decision. It is proper to be {increasing} the second numeral {prev_x}.",
    "You went with {action} starting at {prev_x}. Appropriately, you are {increasing} the second quantity {prev_x}, which is correct.",
    "Choosing {action} from {prev_x} was your action. You are right in {increasing} the second number {prev_x}.",
    "Your pick was {action} from {prev_x}. Indeed, it’s correct to be {increasing} the second digit {prev_x}.",
    "You’ve selected {action} from {prev_x}. Rightly so, {increasing} the second value {prev_x} is the correct course.",
    "The decision to go with {action} from {prev_x} was yours. It’s accurate to be {increasing} the second number {prev_x}."
)

hn_feedback_dim1_template = "You chose {action} from {prev_x}. {increasing} the first number {prev_x} is incorrect."
hn_feedback_dim1 = (
    "Your selection was {action}, starting with {prev_x}. However, {increasing} the initial figure {prev_x} is not the right move.",
    "You decided upon {action} from the point of {prev_x}. Yet, {increasing} the first digit {prev_x} is a mistake.",
    "Opting for {action} from {prev_x} was your choice, but it is wrong to {increasing} the first value {prev_x}.",
    "You went with {action} beginning at {prev_x}, but {increasing} the first number {prev_x} is a misstep.",
    "Your choice was {action} from {prev_x}. Nevertheless, it is incorrect to {increasing} the initial numeral {prev_x}.",
    "Choosing {action} starting from {prev_x} was your action. However, {increasing} the first quantity {prev_x} is not correct.",
    "You picked {action} from {prev_x}, but it's not right to {increasing} the first figure {prev_x}.",
    "Selecting {action} from {prev_x} was your decision, but {increasing} the initial value {prev_x} is erroneous.",
    "You selected {action} from {prev_x}. It’s mistaken to {increasing} the first number {prev_x}, though.",
    "The decision to go with {action} from {prev_x} was yours, but {increasing} the first digit {prev_x} is the wrong approach."
)

hn_feedback_dim2_template = "You chose {action} from {prev_x}. {increasing} the second number {prev_x} is incorrect."
hn_feedback_dim2 = (
    "Your selection was {action}, starting with {prev_x}. However, {increasing} the subsequent number {prev_x} is not the right move.",
    "You decided upon {action} from the point of {prev_x}. Yet, {increasing} the second number {prev_x} is a mistake.",
    "Opting for {action} from {prev_x} was your choice, but it is wrong to {increasing} the second value {prev_x}.",
    "You went with {action} beginning at {prev_x}, but {increasing} the second number {prev_x} is a misstep.",
    "Your choice was {action} from {prev_x}. Nevertheless, it is incorrect to {increasing} the second numeral {prev_x}.",
    "Choosing {action} starting from {prev_x} was your action. However, {increasing} the second quantity {prev_x} is not correct.",
    "You picked {action} from {prev_x}, but it's not right to {increasing} the second number {prev_x}.",
    "Selecting {action} from {prev_x} was your decision, but {increasing} the second value {prev_x} is erroneous.",
    "You selected {action} from {prev_x}. It’s mistaken to {increasing} the second number {prev_x}, though.",
    "The decision to go with {action} from {prev_x} was yours, but {increasing} the second number {prev_x} is the wrong approach."
)

fp_feedback_dim1_template = "You chose {x}. Output a {smaller} number than the first number of {x} to minimize y."
fp_feedback_dim1 = (
    "Your selection was {x}. To minimize y, provide a {smaller} number than the initial digit of {x}.",
    "You’ve opted for {x}. For the purpose of minimizing y, you should produce a {smaller} value than the first number in {x}.",
    "Your choice was {x}. Now, issue a {smaller} number than {x}'s first numeral to reduce y.",
    "You decided on {x}. In order to minimize y, generate a {smaller} figure than the first number of {x}.",
    "{x} was your pick. To achieve a minimal y, output a number that is {smaller} than the first in {x}.",
    "You went with {x}. To make y as small as possible, provide a {smaller} number than {x}'s initial one.",
    "Having chosen {x}, now output a {smaller} number than the first of {x} to bring y down.",
    "Since your selection is {x}, now aim for a {smaller} number than the initial figure in {x} to minimize y.",
    "You have selected {x}. To minimize y, ensure to output a {smaller} number than what starts off {x}.",
    "Your choice fell on {x}. Now, to minimize y, you should opt for a number that is {smaller} than the first in {x}."
)

fp_feedback_dim2_template = "You chose {x}. Output a {smaller} number than the second number of {x} to minimize y."
fp_feedback_dim2 = (
    "Your selection was {x}. To minimize y, produce a {smaller} number than the second digit of {x}.",
    "Having picked {x}, now issue a {smaller} number than {x}'s second figure to reduce y.",
    "You decided on {x}. In your quest to minimize y, output a {smaller} number than the second in {x}.",
    "Your choice fell on {x}. To achieve a lower y, you should select a {smaller} number than the second one of {x}.",
    "You opted for {x}. Minimize y by outputting a {smaller} number than the second digit in {x}.",
    "Given that you’ve chosen {x}, aim to output a {smaller} number than the second numeral of {x} to minimize y.",
    "You went with {x}. Now, to bring y down, provide a {smaller} number than the second of {x}.",
    "Since you selected {x}, to minimize y, choose a {smaller} number than the second figure in {x}.",
    "Your pick was {x}. To make y as small as possible, issue a {smaller} number than {x}'s second.",
    "You’ve chosen {x}. For minimizing y, it's essential to output a {smaller} number than the second value in {x}."
)

fn_feedback_dim1_template = "You chose {x}. Do not output a {smaller} number than the first number in {x} to minimize y."
fn_feedback_dim1 = (
    "Your selection was {x}. Avoid issuing a {smaller} number than the initial digit of {x} in your effort to minimize y.",
    "Having picked {x}, ensure not to produce a {smaller} number than the first figure in {x} as you work to minimize y.",
    "You decided on {x}. In order to minimize y, refrain from outputting a {smaller} number than the first in {x}.",
    "Your choice fell on {x}. To achieve a reduced y, you should not select a {smaller} number than the first one of {x}.",
    "You opted for {x}. For minimizing y, it's crucial not to choose a {smaller} number than the initial digit of {x}.",
    "Given your choice of {x}, to minimize y, do not provide a {smaller} number than the first numeral of {x}.",
    "You’ve selected {x}. As you aim to minimize y, make sure not to output a {smaller} number than the first in {x}.",
    "You picked {x}. To ensure y is minimized, do not issue a {smaller} number than {x}'s initial figure.",
    "Your selection was {x}, and to minimize y, it is important not to produce a {smaller} number than the first value in {x}.",
    "You went with {x}. For the purpose of minimizing y, you should avoid outputting a {smaller} number than the first digit of {x}."
)

fn_feedback_dim2_template = "You chose {x}. Do not output a {smaller} number than the second number in {x} to minimize y."
fn_feedback_dim2 = (
"Your selection was {x}. Avoid issuing a {smaller} number than the second digit of {x} to reduce y.",
    "Having picked {x}, ensure not to produce a {smaller} number than the second figure in {x} as you aim to minimize y.",
    "You decided on {x}. In your quest to make y smaller, refrain from outputting a {smaller} number than the second in {x}.",
    "Your choice fell on {x}. To achieve a minimal y, you should not select a {smaller} number than the second one of {x}.",
    "You’ve opted for {x}. For minimizing y, it's essential not to choose a {smaller} number than the second digit of {x}.",
    "Given your choice of {x}, to minimize y, do not provide a {smaller} number than the second numeral of {x}.",
    "You selected {x}. As you work to minimize y, make sure not to output a {smaller} number than the second in {x}.",
    "You picked {x}. To ensure y is minimized, do not issue a {smaller} number than {x}'s second figure.",
    "Your selection was {x}, and to minimize y, it is important not to produce a {smaller} number than the second value in {x}.",
    "You went with {x}. For the purpose of minimizing y, you should avoid outputting a {smaller} number than the second digit of {x}."
)