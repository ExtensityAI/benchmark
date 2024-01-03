import parso
import difflib
import numpy as np

from pathlib import Path


def normalize(norm_score):
    def _func(score):
        score = score / norm_score
        score = np.minimum(score, 1.0)
        return score
    return _func


# used as a primitive function for the Symbol class
def ast_similarity(self, tree2):
    tree1 = self.value

    def parse_file_to_ast(filename):
        with open(filename, "r") as file:
            source = file.read()
        # Parse the source code with parso, which is more tolerant to errors
        return parso.parse(source), source

    # Check if the input is a file path or an AST
    if (isinstance(tree1, str) or isinstance(tree1, Path)) and os.path.exists(tree1):
        tree1, _ = parse_file_to_ast(tree1)
    # Assume that the input is a source code string
    elif isinstance(tree1, str):
        tree1 = parso.parse(tree1)

    if (isinstance(tree2, str) or isinstance(tree2, Path)) and os.path.exists(tree2):
        tree2, _ = parse_file_to_ast(tree2)
    elif isinstance(tree2, str):
        tree2 = parso.parse(tree2)

    def tree_to_str(node):
        # Generate a string representation of the nodes in the parse tree
        if node.type == 'endmarker':  # Skip the end marker that parso adds
            return ""
        children_str = ''.join(tree_to_str(child) for child in node.children) if hasattr(node, 'children') else ""
        return f"{node.type}({children_str})"

    # Convert parse trees to string representations
    str1 = tree_to_str(tree1)
    str2 = tree_to_str(tree2)

    # Use SequenceMatcher to calculate the ratio of similarity
    matcher = difflib.SequenceMatcher(None, str1, str2)
    similarity = matcher.ratio()
    return similarity
