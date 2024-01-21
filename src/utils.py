import os
import copy
import parso
import difflib
import string
import numpy as np

from pathlib import Path

from symai import Symbol


success_score = {'scores': [1.0]}
mock_score    = copy.deepcopy(success_score)
mock_score.update({'mock': True})
MOCK_RETURN = (True, mock_score)


def bool_success(res):
    return {'scores': [1.0 if res else 0.0]}


def normalize_score(base_score, rand_score, eps=1e-8):
    def _func(score):
        nonlocal base_score, rand_score
        base_score = base_score.value if isinstance(base_score, Symbol) else base_score
        rand_score = rand_score.value if isinstance(rand_score, Symbol) else rand_score
        score      = score.value if isinstance(score, Symbol) else score
        # Ensure that the baseline score is always higher or equal to the random score
        z       = 1.0 / np.maximum(base_score - rand_score, eps)
        z_rand  = rand_score * z
        score   = score * z - z_rand
        # Do not allow negative scores
        res     = np.clip(score, 0.0, 1.0)
        if isinstance(res, np.ndarray):
            res = res.item()
        return res
    return _func


# set the default normalization function
normalize = normalize_score
# use all printable characters as a random sequence
RANDOM_SEQUENCE = string.printable
# reversed random sequence
REVERSED_RANDOM_SEQUENCE = RANDOM_SEQUENCE[::-1]


# general metric for similarity measure
METRIC = 'cosine'
# kernel for distance measure
KERNEL = 'gaussian'


def similarity_measure(self, other, normalize=None):
    # Measure the similarity between two symbols
    val = self.similarity(other, metric=METRIC, normalize=normalize)
    if METRIC == 'cosine':
        # account for the fact that cosine similarity is bounded between -1 and 1
        # by normalizing the score to be between 0 and 1
        res = np.clip(val, 0.0, 1.0)
        if isinstance(res, np.float64):
            res = res.item()
        return res
    if isinstance(res, np.float64):
        res = res.item()
    return val


def distance_measure(self, other, normalize=None):
    # Measure the similarity between two symbols
    res = self.distance(other, kernel=KERNEL, normalize=normalize)
    if isinstance(res, np.float64):
        res = res.item()
    return res


def frechet_measure(self, other, normalize=None):
    # Measure the similarity between two symbols
    sigma1 = np.cov(self.embedding, rowvar=False)
    sigma2 = np.cov(other.embedding, rowvar=False)
    res    = self.distance(other, kernel='frechet', normalize=normalize, sigma1=sigma1, sigma2=sigma2)
    if isinstance(res, np.float64):
        res = res.item()
    return res


# set the default measure
measure = distance_measure


def embedding_mean(self, axis=0):
    # Compute the mean of the embedding
    res = np.mean(self.embedding, axis=axis)
    return Symbol(res)


def cross_validation_score(self, folds=2):
    # Compute the cross validation score
    embeddings = self.embedding
    assert len(embeddings.shape) == 2, "Embeddings must be a 2D array"
    assert embeddings.shape[0] >= folds, "Number of folds must be less than the number of embeddings"
    # permute indices for cross validation
    indices = np.random.permutation(embeddings.shape[0])
    # compute leave-one-out cross validation score
    scores = []
    for i in range(folds):
        # leave out the i-th embedding
        test_idx    = indices[i]
        test_sample = embeddings[test_idx]
        # train on the rest of the embeddings
        train_idx    = np.delete(indices, i)
        train_sample = embeddings[train_idx]
        # compute the mean of the training sample
        train_mean = np.mean(train_sample, axis=0)
        # compute the distance between the test sample and the training mean
        score = Symbol(train_mean).measure(Symbol(test_sample))
        scores.append(score)
    return Symbol(np.mean(scores))


def parse_file_to_ast(filename):
    with open(filename, "r") as file:
        source = file.read()
    # Parse the source code with 9807parso, which is more tolerant to errors
    return parso.parse(source), source


def tree_to_str(node):
    # Generate a string representation of the nodes in the parse tree
    if node.type == 'endmarker':  # Skip the end marker that parso adds
        return ""
    children_str = ''.join(tree_to_str(child) for child in node.children) if hasattr(node, 'children') else ""
    return f"{node.type}({children_str})"


def rand_ast_measure(tree, random_sequence=RANDOM_SEQUENCE):
    if (isinstance(tree, str) or isinstance(tree, Path)) and os.path.exists(tree):
        tree, _ = parse_file_to_ast(tree)
    elif isinstance(tree, str):
        tree = parso.parse(tree)
    elif isinstance(tree, Symbol):
        tree = parso.parse(str(tree))
    # Convert parse trees to string representations
    str_     = tree_to_str(tree)
    # Generate a random parse tree
    random_tree  = parso.parse(random_sequence)
    randstr      = tree_to_str(random_tree)
    # Random string similarity
    matcher         = difflib.SequenceMatcher(None, str_, randstr)
    rand_similarity = matcher.ratio()
    return rand_similarity


# used as a primitive function for the Symbol class
def ast_measure(self, tree2, normalize=None):
    tree1 = self.value

    # Check if the input is a file path or an AST
    if (isinstance(tree1, str) or isinstance(tree1, Path)) and os.path.exists(tree1):
        tree1, _ = parse_file_to_ast(tree1)
    # Assume that the input is a source code string
    elif isinstance(tree1, str):
        tree1    = parso.parse(tree1)

    if (isinstance(tree2, str) or isinstance(tree2, Path)) and os.path.exists(tree2):
        tree2, _ = parse_file_to_ast(tree2)
    elif isinstance(tree2, str):
        tree2 = parso.parse(tree2)
    elif isinstance(tree2, Symbol):
        tree2 = parso.parse(str(tree2))

    # Convert parse trees to string representations
    str1    = tree_to_str(tree1)
    str2    = tree_to_str(tree2)

    # Use SequenceMatcher to calculate the ratio of similarity
    matcher     = difflib.SequenceMatcher(None, str1, str2)
    similarity  = matcher.ratio()

    # Normalize the similarity score
    if normalize is not None:
        # Normalize the similarity score
        similarity = normalize(similarity)
    return similarity
