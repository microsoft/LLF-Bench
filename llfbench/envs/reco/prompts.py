movie_instruction_template = "You are a helpful assistant trying to recommend movies to your users according to what they want."
movie_instruction = (
    "As a helpful assistant, your aim is to suggest films that align with your users' preferences.",
    "Your role as an accommodating aide involves proposing movies that match your users' desires.",
    "You’re an obliging assistant, dedicated to advising your users on movies that cater to their tastes.",
    "Being a supportive assistant, you strive to guide users to movies they'll enjoy based on their wishes.",
    "As an assistive aide, you're focused on providing movie recommendations that resonate with what your users seek.",
    "You take on the role of a helpful assistant, selecting movies that reflect the interests of your users.",
    "Your mission as a helpful assistant is to present movie options tailored to your users' requests.",
    "As an attentive assistant, you make it your job to endorse movies that your users express an interest in."
)

no_rec_r_neg_template = "You didn't recommend anything to me."
no_rec_r_neg = (
    "You haven’t provided any recommendations to me.",
    "No suggestions have been made on your part.",
    "You failed to suggest any titles to me.",
    "I haven't received any movie or TV show recommendations from you.",
    "It seems like you've overlooked recommending any movies or shows to me.",
    "You've not offered any recommendations for movies or TV series.",
    "No movie or TV show recommendations have come my way from you.",
    "You haven't made any movie or TV suggestions to me."
)

hallucination_r_pos_template = "I can find all the recommended {movie}s, nice!"
hallucination_r_pos = (
    "All of the suggested {movie}s are available, which is great!",
    "I've located every one of the recommended {movie}s, awesome!",
    "I managed to find all the {movie}s you recommended, nice going!",
    "Every {movie} that was recommended is findable online, splendid!",
    "It's great that all the recommended {movie}s are accessible!",
    "I have access to all the {movie}s that were recommended, which is fantastic!",
    "All the {movie}s on the recommended list can be found, wonderful!",
    "I'm pleased to find each of the recommended {movie}s available online, nice!"
)

hallucination_r_neg_template = "I can't find some of the recommended {movie}s on the internet."
hallucination_r_neg = (
    "Some of the suggested {movie}s are not discoverable on the internet.",
    "A few of the recommended {movie}s seem to be missing online.",
    "I'm unable to locate a number of the recommended {movie}s on the web.",
    "Several of the {movie}s you recommended aren't available on the internet.",
    "I'm having trouble finding some of the {movie}s you suggested online.",
    "Not all of the recommended {movie}s can be found on the internet.",
    "Some of the {movie}s that were recommended are elusive online.",
    "A portion of the recommended {movie}s appear to be absent from the internet."
)

hallucination_hp_template ="I can find these {movie}s on the internet:"
hallucination_hp = (
    "These {movie}s are available on the internet:",
    "I'm able to locate these {movie}s online:",
    "I have found these {movie}s on the web:",
    "These {movie}s are accessible to me on the internet:",
    "The internet has these {movie}s listed:",
    "I can locate these {movie}s somewhere on the internet:",
    "These {movie}s are present and accounted for online:",
    "I've found these {movie}s available for viewing on the internet:"
)

hallucination_hn_template = "I can't find these {movie}s on the internet:"
hallucination_hn = (
    "These {movie}s aren't available on the internet:",
    "I'm unable to locate these {movie}s online:",
    "These {movie}s are missing from the internet:",
    "I’ve searched online but can't locate these {movie}s:",
    "The internet doesn’t seem to have these {movie}s:",
    "These {movie}s are not to be found on the web:",
    "I can't seem to find these {movie}s anywhere online:",
    "These {movie}s are nowhere to be found on the internet:"
)

hallucination_fp_template = "Recommend {movie}s that I can find online, like:"
hallucination_fp = (
    "Suggest {movie}s that are available online, such as:",
    "Please propose {movie}s that can be located online, for example:",
    "I would appreciate recommendations for {movie}s that are accessible online, like:",
    "Offer up {movie}s I can stream or download online, including:",
    "Focus on recommending {movie}s that are available for online viewing, such as:",
    "Provide suggestions for {movie}s that I can find on the internet, for instance:",
    "Endorse {movie}s that are readily available online, like:",
    "I’m looking for {movie}s that can be found online, such as:"
)

hallucination_fn_template = "Do not recommend {movie}s that I can't find online, like:"
hallucination_fn = (
    "Avoid suggesting {movie}s that are unavailable online, such as:",
    "Please don’t propose {movie}s that aren’t accessible online, for example:",
    "I’d prefer not to get recommendations for {movie}s that can't be located online, like:",
    "Refrain from recommending {movie}s that I can't stream or download, including:",
    "Please exclude {movie}s from your recommendations if they're not online, like:",
    "Ensure not to endorse {movie}s that are not available for online viewing, such as:",
    "Steer clear of advising {movie}s that aren't online, for instance:",
    "I would like to avoid recommendations for {movie}s that are not found online, like:"
)

type_r_pos_template = "What you recommended are {movie}s, nice!"
type_r_pos = (
    "The recommendations you made are {movie}s, great!",
    "You've recommended {movie}s, which is awesome!",
    "The suggestions you provided are {movie}s, nice one!",
    "What's on the recommendation list are {movie}s, fantastic!",
    "Your recommended picks are {movie}s, wonderful!",
    "The titles you suggested are indeed {movie}s, excellent!",
    "Nice, the options you suggested are {movie}s!",
)

type_r_neg_template = "The recommended items are not all {movie}s."
type_r_neg = (
    "Not all of the suggested items are {movie}s.",
    "The recommendations include more than just {movie}s.",
    "Not every recommended item is a {movie}.",
    "The list of recommended items isn't composed solely of {movie}s.",
    "Among the recommended items, there are some non-{movie}s as well.",
    "The suggested items are a mix, not exclusively {movie}s.",
    "There's a variety in the recommendations, not just {movie}s.",
    "The recommendations span beyond just {movie}s."
)

type_hp_template = "These items are indeed all {movie}s:"
type_hp = (
    "Indeed, each of these items is a {movie}:",
    "All of these items are, in fact, {movie}s:",
    "It's confirmed that every item here is a {movie}:",
    "Without exception, all these items are {movie}s:",
    "Each one of the items listed is definitely a {movie}:",
    "True, the entire selection here consists of {movie}s:",
    "Yes, all these items are categorized as {movie}s:",
    "Certainly, these items are all {movie}s:"
)

type_hn_template = "These items are not all {movie}s:"
type_hn = (
    "Not every one of these items is a {movie}.",
    "These items include more than just {movie}s.",
    "Not all items listed here are {movie}s.",
    "The items here aren't exclusively {movie}s.",
    "Among these items, not all are {movie}s.",
    "There's a variety among these items; they're not all {movie}s.",
    "These items are a mix, with not every one being a {movie}.",
    "The collection here extends beyond just {movie}s."
)

type_fp_template = "Recommend {movie}s, like"
type_fp = (
    "Suggest {movie}s, such as",
    "Propose {movie}s, for example",
    "Advise on {movie}s, similar to",
    "Offer recommendations for {movie}s, like",
    "Give examples of {movie}s, including",
    "Provide me with {movie}s, akin to",
)

type_fn_template = "Do not recommend items that are not {movie}s, like"
type_fn = (
    "Avoid suggesting items that aren't {movie}s, such as",
    "Please don't propose items if they're not {movie}s, like",
    "Refrain from recommending non-{movie} items, for instance",
    "Steer clear of advising items that don't fall under {movie}s, including",
    "Eschew offering recommendations for items not related to {movie}s, similar to",
    "Exclude non-{movie} items from suggestions, akin to",
    "Keep away from indicating items that are not {movie}s, resembling",
    "Omit items that aren't {movie}s from recommendations, such as"
)

genre_r_pos_template = "The recommended {movie}s are all {action_comedy_movie}, nice!"
genre_r_pos = (
    "All of the suggested {movie}s are {action_comedy_movie}, great!",
    "Every one of the recommended {movie}s is an {action_comedy_movie}, which is awesome!",
    "The {movie}s you've recommended are exclusively {action_comedy_movie}, nice one!",
    "It's cool that the recommended {movie}s are all {action_comedy_movie}!",
    "Nice to see that each recommended {movie} is a {action_comedy_movie}!",
    "All the {movie}s on the recommendation list are {action_comedy_movie}, wonderful!",
    "Delighted that the recommended {movie}s are all in the {action_comedy_movie} genre, excellent!",
    "The entire list of recommended {movie}s consists of {action_comedy_movie}, which is fantastic!"
)

genre_r_neg_template = "The recommendations are not all {action_comedy_movie}s."
genre_r_neg = (
    "Not every recommendation is an {action_comedy_movie}.",
    "The suggested titles aren't exclusively {action_comedy_movie}s.",
    "The recommendations include more than just {action_comedy_movie}s.",
    "Not all of the recommended picks are {action_comedy_movie}s.",
    "There's a variety in the recommendations, not limited to {action_comedy_movie}s.",
    "The recommendations span a broader range than just {action_comedy_movie}s.",
    "Among the recommended titles, you'll find more than {action_comedy_movie}s.",
    "The list of suggestions contains more than solely {action_comedy_movie}s."
)

genre_hp_template = "These {movie}s are indeed {action_comedy}:"
genre_hp = (
    "Indeed, these {movie}s are categorized as {action_comedy}:",
    "Each of these {movie}s is truly an {action_comedy}:",
    "It is confirmed that these {movie}s fall under the {action_comedy} genre:",
    "Without a doubt, these {movie}s are {action_comedy}:",
    "Certainly, all these {movie}s qualify as {action_comedy}:",
    "These {movie}s can be rightly classified as {action_comedy}:",
    "Yes, these {movie}s are representative of the {action_comedy} genre:",
    "Each one of these {movie}s is definitively an {action_comedy}:"
)

genre_hn_template = "These {movie}s are not {action_comedy}:"
genre_hn = (
    "These {movie}s do not fall within the {action_comedy} genre:",
    "These {movie}s aren't classified as {action_comedy}:",
    "The {movie}s listed here are not categorized as {action_comedy}:",
    "These {movie}s cannot be considered {action_comedy}:",
    "None of these {movie}s are {action_comedy}:",
    "These {movie}s are outside the {action_comedy} category:",
    "The {movie}s presented here do not belong to the {action_comedy} genre:",
    "It turns out that these {movie}s are not {action_comedy}:"
)

genre_fp_template = "Recommend {movie}s that are {action_comedy}, like"
genre_fp = (
    "Suggest {movie}s falling into the {action_comedy} genre, such as",
    "Propose {movie}s which are {action_comedy}, for example",
    "Advise on {movie}s with an {action_comedy} theme, similar to",
    "Give recommendations for {action_comedy} {movie}s, like",
    "Offer a list of {movie}s that are {action_comedy}, including",
    "Identify {movie}s characterized as {action_comedy}, akin to",
    "Select {movie}s that exemplify the {action_comedy} genre, as in",
    "Choose {movie}s that are in the {action_comedy} category, like"
)

genre_fn_template = "Do not recommend {movie}s that are not {action_comedy}, not like"
genre_fn = (
    "Avoid suggesting {movie}s if they're outside the {action_comedy} genre, unlike",
    "Please refrain from recommending {movie}s that don't fit the {action_comedy} category, not similar to",
    "Steer clear of proposing {movie}s that aren't {action_comedy}, in contrast to",
    "Exclude {movie}s that are not {action_comedy} from your suggestions, not as in",
    "Do not advise on {movie}s lacking {action_comedy} elements, not resembling",
    "Omit {movie}s that do not classify as {action_comedy}, not akin to",
    "Eschew offering {movie}s that diverge from the {action_comedy} type, not comparable to",
    "Keep away from {movie}s that aren't representative of {action_comedy}, not following the example of"
)

year_r_pos_template = "The recommended movies are all from the {correct_years}, great!"
year_r_pos = (
    "All the suggested movies hail from the {correct_years}, which is fantastic!",
    "Every one of the recommended films is from the {correct_years}, awesome!",
    "The movies you've recommended are exclusively from the {correct_years}, excellent!",
    "It’s great that all the recommended movies belong to the {correct_years}!",
    "Delighted to see that each movie recommended comes from the {correct_years}!",
    "The entire selection of recommended movies originates from the {correct_years}, wonderful!",
    "All the films on the recommended list are from the {correct_years}, splendid!",
    "The recommended list features movies solely from the {correct_years}, which is perfect!"
)

year_r_neg_template = "The recommended {movie}s are not from the {correct_years}."
year_r_neg = (
    "The suggested {movie}s don't all hail from the {correct_years}.",
    "Some of the recommended {movie}s do not originate from the {correct_years}.",
    "The {movie}s you've recommended aren't all from the {correct_years}.",
    "It turns out the recommended {movie}s are not all from the {correct_years}.",
    "The list of recommended {movie}s are not all from the {correct_years}.",
    "Regrettably, some of the recommended {movie}s fall outside of the {correct_years}.",
    "Some of the recommended {movie}s are unfortunately not all from the specified {correct_years}."
)

year_hp_template = "These {movie}s are indeed from the {correct_years}:"
year_hp = (
    "Indeed, these {movie}s originate from the {correct_years}:",
    "These {movie}s are certainly from the {correct_years}:",
    "It's confirmed that these {movie}s belong to the {correct_years}:",
    "True, these {movie}s are from the {correct_years}:",
    "Without question, these {movie}s date back to the {correct_years}:",
    "Affirmative, these {movie}s were produced in the {correct_years}:",
    "These {movie}s definitely represent the {correct_years}:",
    "Yes, these {movie}s are from the era of the {correct_years}:"
)

year_hn_template = "These {movie}s are not all from the {correct_years}:"
year_hn = (
    "Not every one of these {movie}s originates from the {correct_years}:",
    "These {movie}s don't all hail from the {correct_years}:",
    "A number of these {movie}s fall outside the {correct_years}:",
    "Some of these {movie}s are not dated within the {correct_years}:",
    "The release years of these {movie}s don't all match the {correct_years}:",
    "These {movie}s aren't exclusively from the {correct_years}:",
    "It's not the case that all these {movie}s were produced in the {correct_years}:",
    "Each of these {movie}s does not necessarily correspond to the {correct_years}:"
)

year_fp_template = "Recommend {movie}s that are from {correct_years}, like"
year_fp = (
    "Suggest {movie}s dating back to {correct_years}, such as",
    "Propose {movie}s from the era of {correct_years}, for instance",
    "Advise on {movie}s that originate from {correct_years}, similar to",
    "Identify {movie}s that were released during {correct_years}, like",
    "Put forward {movie}s corresponding to {correct_years}, including",
    "Select {movie}s representative of {correct_years}, exemplified by",
    "Choose {movie}s which were made in {correct_years}, as in",
    "Present {movie}s from the {correct_years} period, akin to"
)

year_fn_template = "Do not recommend {movie}s that are not from {correct_years}, like"
year_fn = (
    "Avoid suggesting {movie}s that fall outside of {correct_years}, such as",
    "Refrain from proposing {movie}s that weren't made in {correct_years}, for example",
    "Steer clear of recommending {movie}s not produced during {correct_years}, similar to",
    "Exclude {movie}s from the recommendations if they're not from {correct_years}, including",
    "Do not put forward {movie}s that do not date back to {correct_years}, like",
    "Eschew selecting {movie}s that aren't associated with {correct_years}, as in",
    "Omit {movie}s from your suggestions if they are not from the period of {correct_years}, exemplified by",
    "Please do not advise on {movie}s from years other than {correct_years}, such as"
)

child_friendly_r_pos_template = "The recommended {movie}s are all {child_friendly}, awesome!"
child_friendly_r_pos = (
"Every one of the suggested {movie}s is {child_friendly}, which is fantastic!",
    "All the {movie}s on the recommendation list are {child_friendly}, excellent!",
    "Delighted to see that the recommended {movie}s are uniformly {child_friendly}, wonderful!",
    "The {movie}s you’ve recommended are indeed all {child_friendly}, terrific!",
    "It's great to find that each of the {movie}s recommended is {child_friendly}, splendid!",
    "The entire selection of recommended {movie}s is {child_friendly}, how delightful!",
    "Pleased to report that every recommended {movie} is {child_friendly}, superb!"
)

child_friendly_r_neg_template = "The recommended {movie}s are not all {child_friendly}."
child_friendly_r_neg = (
    "Not all of the suggested {movie}s are {child_friendly}.",
    "Some of the recommended {movie}s aren't {child_friendly}.",
    "A few of the recommended {movie}s fall short of being {child_friendly}.",
    "The list of recommended {movie}s includes some that are not {child_friendly}.",
    "Among the recommended {movie}s, not every one is {child_friendly}.",
    "It appears not every {movie} recommended is {child_friendly}.",
    "Not each of the {movie}s on the recommendation list is {child_friendly}.",
    "The recommended {movie}s aren’t all classified as {child_friendly}."
)

child_friendly_hp_template = "These {movie}s are indeed {child_friendly}:"
child_friendly_hp = (
    "Certainly, these {movie}s are {child_friendly}:",
    "It's confirmed that these {movie}s are {child_friendly}:",
    "Absolutely, these {movie}s meet the {child_friendly} criteria:",
    "These {movie}s are, without a doubt, {child_friendly}:",
    "True to form, these {movie}s are {child_friendly}:",
    "Undoubtedly, these {movie}s are {child_friendly}:",
    "These selected {movie}s are acknowledged as {child_friendly}:",
    "Each of these {movie}s is verified as {child_friendly}:"
)

child_friendly_hn_template = "These {movie}s are not {child_friendly}:"
child_friendly_hn = (
    "Not every one of these {movie}s is {child_friendly}:",
    "These {movie}s aren't uniformly {child_friendly}:",
    "Some of these {movie}s do not qualify as {child_friendly}:",
    "Among these {movie}s, some are not {child_friendly}:",
    "A selection of these {movie}s is not {child_friendly}:",
    "These {movie}s vary, with not all being {child_friendly}:",
)

child_friendly_fp_template = "Recommend {movie}s that are {child_friendly}, like"
child_friendly_fp = (
    "Suggest {movie}s which are classified as {child_friendly}, such as",
    "Propose a list of {movie}s that carry the {child_friendly} label, for instance",
    "Provide recommendations for {movie}s that have a {child_friendly} rating, like",
    "Advise on {movie}s deemed {child_friendly}, including",
    "Point me towards {movie}s that are known to be {child_friendly}, similar to",
    "Identify {movie}s appropriate for the {child_friendly} category, exemplified by",
    "Curate a selection of {movie}s that fit the {child_friendly} criteria, akin to",
    "Highlight {movie}s that are rated as {child_friendly}, such as"
)

child_friendly_fn_template = ""
child_friendly_fn = (
    "Avoid suggesting {movie}s which lack a {child_friendly} designation, such as",
    "Refrain from recommending {movie}s that don’t meet the {child_friendly} standard, like",
    "Steer clear of {movie}s not recognized as {child_friendly}, for instance",
    "Exclude {movie}s from the list if they are not {child_friendly}, exemplified by",
    "Omit {movie}s that do not have a {child_friendly} rating, similar to",
    "Please do not suggest {movie}s if they’re not {child_friendly}, such as",
    "Eschew proposing {movie}s that aren’t considered {child_friendly}, including",
    "Bypass {movie}s that aren’t labeled as {child_friendly}, like"
)