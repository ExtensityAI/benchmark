from symai import Expression
from symai.components import Sequence, Parallel
from symai.extended import Conversation
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor


class Context(Expression):
    def __init__(self, description: str = None, **kwargs):
        super().__init__(**kwargs)
        self._description    = description or ''
        self.post_processors = [StripPostProcessor(), CodeExtractPostProcessor()]

    @property
    def description(self):
        return self._description

    @property
    def static_context(self):
        return """[General Context]
Write a scientific paper about the machine learning framework called SymbolicAI which operates on the following principles:
- Symbolic methods
- Sub-symbolic methods
- Neural-symbolic methods
- Probabilistic programming methods
- Cognitive architectures

[Format]
Your output format should be parsable by a LaTeX compiler. All produced content should be enclosed between the \n```latex\n ... \n``` blocks. Do not create document classes or other LaTeX meta commands. Always assume that the document class is already defined. Only produce exactly one latex block with all your content.
The following is an example of your expected output:

[Example]
```latex
\documentclass{{article}}
\begin{{document}}
% TODO: your content here
\end{{document}}
```
Replace the % TODO: with your content and follow the task description below.

{description}
""".format(description=self.description)


class Paper(Expression):
    def __init__(self, *sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = Sequence(*sequence)

    def forward(self, task):
        return self.sequence(task)


class Source(Conversation, Context):
    @property
    def description(self):
        return """[Task]
Summarized the SymbolicAI framework to use it as a conditioning context for a large Language model like GPT-3.
"""

class Method(Context):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.source.auto_print = False

    def forward(self, task):
        summary = self.source(task)
        # update the dynamic context
        self.adapt(context=summary, types=Context)
        return summary

    @property
    def description(self):
        return """[Task]
Your goal is to write the method section which describes the main approach and used principles.
"""


class Cite(Source):
    @property
    def description(self):
        return """[Task]
Write a short two sentence related work summary in the context of the paper.
"""


class RelatedWork(Context):
    def __init__(self, *sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = Parallel(*sequence, sequential=True)

    @property
    def description(self):
        return """[Task]
Write a coherent related work section in the context of the paper and based on the provided citation sources.
"""


class Introduction(Context):
    def __init__(self, *sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = Parallel(*sequence, sequential=True)

    @property
    def description(self):
        return """[Task]
Write a coherent related work section in the context of the paper and based on the provided citation sources.
"""


class Abstract():
    @property
    def description(self):
        return """[Task]
Write an abstract for the paper.
"""


class Title():
    @property
    def description(self):
        return """[Task]
Write a title for the paper.
"""
