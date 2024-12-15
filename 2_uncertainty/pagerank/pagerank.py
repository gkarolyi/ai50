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
    page_probabilities = {key: 0 for key in corpus}

    if len(corpus[page]) == 0:
        for key in page_probabilities:
            page_probabilities[key] = 1 / len(corpus.keys())
    else:
        p = (1 - damping_factor) / len(corpus.keys())
        for key in page_probabilities:
            page_probabilities[key] = p
            if key in corpus[page]:
                page_probabilities[key] += damping_factor / len(corpus[page])

    return page_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {key: 0 for key in corpus}
    sample_key = random.choice(list(corpus.keys()))

    for _ in range(n):
        transition = transition_model(corpus, sample_key, damping_factor)
        pagerank[sample_key] += 1 / n
        sample_key = random.choices(list(transition.keys()), weights=transition.values(), k=1)[0]

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_pagerank = {key: 1 / len(corpus) for key in corpus}
    new_pagerank = {}

    while True:
        for linked_page in corpus:
            choice_probability = 0
            for linking_page in corpus:
                if linked_page in corpus[linking_page]:
                    choice_probability += current_pagerank[linking_page] / len(corpus[linking_page])
                elif len(corpus[linking_page]) == 0:
                    choice_probability += current_pagerank[linking_page] / len(corpus)

            new_pagerank[linked_page] = (damping_factor * choice_probability)
            new_pagerank[linked_page] += (1 - damping_factor) / len(corpus)

        rank_diffs = [new_pagerank[page] - current_pagerank[page] for page in current_pagerank]

        if max(rank_diffs) < 0.001:
            break
        else:
            current_pagerank = new_pagerank.copy()

    return current_pagerank


if __name__ == "__main__":
    main()
