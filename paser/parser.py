import itertools
import time
import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# NonTerminals can be split into
# S         = Sentence
# AdjP      = Adjective Phrase
# AdvP      = Adverb Phrase
# NP        = Noun Phrase
# VP        = Verb phrase
# PP        = Prepositional Phrase

NONTERMINALS = """
S           -> NP VP | S Conj S
AdjP        -> Adj | Adj AdjP
AdvP        -> Adv | Adv AdvP
NP          -> N | Det N | Det AdjP N | NP PP
VP          -> V | V NP | V PP | VP AdvP | AdvP VP | VP Conj VP
PP          -> P NP | P NP PP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # Tokenizing the sentence into words and converting to lowercase
    words = nltk.tokenize.word_tokenize(sentence.lower())

    # Returning list with the words that contains at least one alphabetic character
    return [word for word in words if any(c.isalpha() for c in word)]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    return [
        # Adding the subtree
        subtree for subtree in tree.subtrees()
        # Looping through all subtrees of the subtrees to check that none are nouns
        if subtree.label() == "NP" and not any(subt.label() == "NP" for subt in itertools.islice(subtree.subtrees(), 1, None))
    ]


if __name__ == "__main__":
    main()
