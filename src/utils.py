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


def normalize_score(base_score, rand_score, eps=1e-8):
    def _func(score):
        # Ensure that the baseline score is always higher or equal to the random score
        z       = 1.0 / np.maximum(base_score - rand_score, eps)
        z_rand  = rand_score * z
        score   = score * z - z_rand
        # Do not allow negative scores
        return np.clip(score, 0.0, 1.0)
    return _func


# set the default normalization function
normalize = normalize_score
# use all printable characters as a random sequence
RANDOM_SEQUENCE = string.printable
# reversed random sequence
REVERSED_RANDOM_SEQUENCE = RANDOM_SEQUENCE[::-1]


# general metric for similarity measure
METRIC = 'cosine'
# create a random symbol
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
    # Measure the similarity between two symbols
    return self.distance(other, kernel=KERNEL, normalize=normalize)


measure = distance_measure


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
