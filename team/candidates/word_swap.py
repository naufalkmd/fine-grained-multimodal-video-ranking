"""Find replacement words for `word_swap` negatives from the descriptions themselves.

No word list, dictionary or model: two words are candidates when they appear between the same
neighbouring words across the given descriptions. Very common and very rare words are skipped by
frequency alone. Pass only one split's descriptions, so that splits stay separate.
"""

import re
from collections import Counter, defaultdict

WORD = re.compile(r"[a-z]+")
MIN_DESCRIPTIONS = 5
MAX_DESCRIPTION_SHARE = 0.05
MIN_SHARED_CONTEXTS = 3
NEIGHBOURS = 10


def tokens(text):
    """Lower-case words with their (start, end) positions in the text."""
    return [(match.group(), match.start(), match.end()) for match in WORD.finditer(text.lower())]


def replacement_table(descriptions):
    """Word -> up to NEIGHBOURS words used in the most similar contexts, most similar first."""
    documents = Counter()
    contexts = defaultdict(set)
    for text in descriptions:
        words = [word for word, _, _ in tokens(text)]
        documents.update(set(words))
        for position, word in enumerate(words):
            before = words[position - 1] if position else "<start>"
            after = words[position + 1] if position + 1 < len(words) else "<end>"
            contexts[word].add((before, after))
    limit = MAX_DESCRIPTION_SHARE * len(descriptions)
    usable = {word for word, count in documents.items() if MIN_DESCRIPTIONS <= count <= limit}
    words_by_context = defaultdict(list)
    for word in usable:
        for context in contexts[word]:
            words_by_context[context].append(word)
    shared = defaultdict(Counter)
    for words in words_by_context.values():
        for word in words:
            for other in words:
                if other != word:
                    shared[word][other] += 1
    table = {}
    for word, counts in shared.items():
        scored = [(count / len(contexts[word] | contexts[other]), other)
                  for other, count in counts.items() if count >= MIN_SHARED_CONTEXTS]
        if scored:
            table[word] = [other for _, other in sorted(scored, key=lambda item: (-item[0], item[1]))[:NEIGHBOURS]]
    return table


def match_case(replacement, original):
    return replacement.capitalize() if original[0].isupper() else replacement


def swap_one_word(text, table, rng):
    """Replace one word with a neighbour that the description doesn't already use, or return None."""
    found = tokens(text)
    present = {word for word, _, _ in found}
    options = [(word, start, end) for word, start, end in found
               if any(other not in present for other in table.get(word, ()))]
    if not options:
        return None
    word, start, end = rng.choice(options)
    replacement = rng.choice([other for other in table[word] if other not in present])
    return text[:start] + match_case(replacement, text[start:end]) + text[end:]
