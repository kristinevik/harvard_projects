import copy
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Counting the amount of links
    length_links = len(corpus[page])
    N = len(corpus)

    # If page has links or not, random factor is distributed to all of corpus
    prob = {p: (1 - damping_factor) / N for p in corpus}

    # If the page has links
    if corpus[page]:
        link_prob = damping_factor / len(corpus[page])
        for link in corpus[page]:
            prob[link] += link_prob

    # If the page has no links on it, we pretend it links to every page
    else:
        no_link_prob = damping_factor / N
        for p in corpus:
            prob[p] += no_link_prob

    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create a new dictionary that will hold the pageranks, starting with 0 pagerank for all
    pagerank = {page: 0 for page in corpus}

    # Choose a random page from corpus
    page = random.choice(list(corpus.keys()))

    # Looping through to sample n webpages
    for _ in range(n):
        # First creating an probability distribution
        transition_prob = transition_model(corpus, page, damping_factor)

        # Sampling a link based on the transition probabilities, and now we go to that link
        page = random.choices(list(transition_prob.keys()),
                              weights=transition_prob.values())[0]

        # Add 1/n of the weight to the link we sampled
        pagerank[page] += 1 / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    stop_value = 0.001

    current_rank = {page: 1 / N for page in corpus}

    # Break while loop when changes are less than stop_value
    while True:
        # All pages gets the random factor, plus whatever links other pages has
        new_rank = {p: (1 - damping_factor) / N for p in current_rank}
        for page in corpus:

            for other_page, links in corpus.items():

                # If the page does have links, the page only gets distributed page_rank if it is one of the links
                if links:
                    if page in links:
                        new_rank[page] += damping_factor * \
                            (current_rank[other_page] / len(links))

                # If that page does not have any links, we pretend it is linked to every page
                else:
                    new_rank[page] += damping_factor * \
                        (current_rank[other_page] / N)

        # Normalize so it sums to one
        sum_rank = sum(new_rank.values())

        for page in new_rank:
            new_rank[page] /= sum_rank

        # Check the difference from last iteration
        stop = max(abs(current_rank[all_pages] - new_rank[all_pages])
                   for all_pages in corpus)

        current_rank = copy.deepcopy(new_rank)

        if stop < stop_value:
            break

    return current_rank


if __name__ == "__main__":
    main()
