# Analysis

## Layer 6, Head 12

Example Sentences:
- How do Python [MASK] eat spaghetti code?
    "do" pays strong attention to "eat" and Python pays moderate attention to "eat".

- How do [MASK] networks learn from different data?
    "do" pays strong attention to "learn" and [MASK] pays moderate attention to "learn".

This head seems to focus on the relationship between the helping verb "do" and the main verb (eat/learn). That will help the model with the grammatical structure of a sentence, particulary questions it seems.

In addition, the moderate attention is between the word after "do" to the main verb. This could mean that the head is using this information to find the main topic of the question that should be linked to the action.

## Layer 4, Head 6

Example Sentences:
- I now know how to create AI models that [MASK].
- I have really enjoyed all the learning in this [MASK].

For both sentences, the attention diagram has a strong diagonal line from the beginning to the end of the sentence. That means that every token is has the most attention for the previous token in the sentence. That means that this attention head seems to be focusing on understanding the sequential order of words.




