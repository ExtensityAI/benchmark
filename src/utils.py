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


def _is_numpy_array(val):
    return isinstance(val, np.float64) or \
           isinstance(val, np.float32) or \
           isinstance(val, np.float16) or \
           isinstance(val, np.float_)  or \
           isinstance(val, np.ndarray)


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
        return Symbol(res)
    return _func


# set the default normalization function
normalize = normalize_score
# use all printable characters as a random sequence
RANDOM_SEQUENCE = string.printable
# reversed random sequence
REVERSED_RANDOM_SEQUENCE = RANDOM_SEQUENCE[::-1]
# some random response
RANDOM_RESPONSE = "As a worthless AI Mockup model, I cannot provide you with any meaningful response. I am sorry. Please try again later."
# the list of all randomness
RANDOMNESS = [RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE, RANDOM_RESPONSE]


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
        return np.clip(val, 0.0, 1.0)
    return val


def distance_measure(self, other, normalize=None):
    # Measure the distance between two symbol distributions
    return self.distance(other, kernel=KERNEL, normalize=normalize)


def frechet_measure(self, other, normalize=None):
    # Measure the distance between two symbol distributions
    sigma1 = np.cov(self.embedding, rowvar=False)
    sigma2 = np.cov(other.embedding, rowvar=False)
    return self.distance(other, kernel='frechet', normalize=normalize, sigma1=sigma1, sigma2=sigma2)


# set the default measure
def measure(self, other, normalize=None):
    # use distance measure as the default measure
    res = self.distance(other, kernel=KERNEL, normalize=normalize)
    # return a Symbol
    if not isinstance(res, Symbol):
        res = Symbol(res)
    return res


def embedding_mean(self, axis=0):
    # Compute the mean of the embedding
    res = np.mean(self.embedding, axis=axis)
    return Symbol(res)


def cross_validation_score(self, min_samples=2):
    # Compute the cross validation score
    embeddings = self.embedding
    assert len(embeddings.shape) == 2, "Embeddings must be a 2D array"
    assert embeddings.shape[0] >= min_samples, "There must be at least two embeddings to perform a (cross) validation"
    # permute indices of embeddings shape[0] to all possible combinations
    indices  = np.random.permutation(embeddings.shape[0])
    # compute leave-one-out cross validation score
    scores   = []
    # if there are only two embeddings, then we can only do one fold
    range_   = 1 if embeddings.shape[0] == 2 else embeddings.shape[0]
    for i in range(range_):
        # leave out the i-th embedding
        test_idx       = indices[i]
        test_sample    = embeddings[test_idx]
        if len(test_sample.shape) == 1:
            test_mean  = test_sample
        else:
            test_mean  = np.mean(test_sample, axis=0)
        # use the rest of the embeddings as the training sample
        train_idx      = np.delete(indices, i)
        train_sample   = embeddings[train_idx]
        # compute the mean of the training sample
        if len(train_sample.shape) == 1:
            train_mean = train_sample
        else:
            train_mean = np.mean(train_sample, axis=0)
        assert train_mean.shape == test_mean.shape, "Train and test mean must have the same shape"
        # compute the distance between the test sample and the training mean
        score = Symbol(train_mean).measure(Symbol(test_sample))
        scores.append(score.value)
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


def rand_ast_measure(tree, random_sequence=RANDOMNESS):
    if (isinstance(tree, str) or isinstance(tree, Path)) and os.path.exists(tree):
        tree, _ = parse_file_to_ast(tree)
    elif isinstance(tree, str):
        tree = parso.parse(tree)
    elif isinstance(tree, Symbol):
        tree = parso.parse(str(tree))
    # Convert parse trees to string representations
    str_     = tree_to_str(tree)
    # Generate a random parse tree
    score = []
    for rand_seq in random_sequence:
        random_tree  = parso.parse(rand_seq)
        randstr      = tree_to_str(random_tree)
        # Random string match
        matcher         = difflib.SequenceMatcher(None, str_, randstr)
        rand_match  = matcher.ratio()
        score.append(rand_match)
    mean_match = np.mean(score).item()
    return mean_match


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

    # Use SequenceMatcher to calculate the ratio of matching characters
    matcher     = difflib.SequenceMatcher(None, str1, str2)
    match_ratio = matcher.ratio()

    # Normalize the match score
    if normalize is not None:
        # Normalize the match score
        match_ratio = normalize(match_ratio)
    return match_ratio
