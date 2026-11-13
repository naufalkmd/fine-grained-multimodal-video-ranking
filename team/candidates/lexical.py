"""Word-overlap similarity between descriptions, corrected for hubs, for `lexical` negatives.

Plain TF-IDF nearest neighbours concentrate on a few generic descriptions: in the 834-episode
validation split one description was the negative of 28 others while 103 were never used, and a
text-only classifier could spot negatives from that skew (score 0.485). CSLS (Conneau et al.,
2018) subtracts each description's mean similarity to its k nearest neighbours, which spreads the
picks and brings the text-only score to chance. Uses only the descriptions; no model scores.
"""

import numpy as np

HUB_NEIGHBOURS = 10


def lexical_similarity(descriptions, k=HUB_NEIGHBOURS):
    """CSLS-corrected TF-IDF cosine similarity; the diagonal is -inf so no text matches itself."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectors = TfidfVectorizer().fit_transform(descriptions)
    similarity = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, len(descriptions) - 1)
    hubness = np.sort(similarity, axis=1)[:, -k:].mean(axis=1)
    return 2 * similarity - hubness[None, :] - hubness[:, None]
