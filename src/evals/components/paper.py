from symai import Function
from symai.components import Sequence, Parallel
from symai.extended import Conversation
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor


PAPER_STATIC_CONTEXT = """[General Context]
Write a scientific paper about the machine learning framework called SymbolicAI which operates on the following principles:
- Symbolic methods
- Sub-symbolic methods
- Neural-symbolic methods
- Probabilistic programming methods
- Cognitive architectures
Be precise in your writing and follow a scientific style. Do not use any colloquial language. However, formulate simple and understandable sentences.

[Format]
Your output format should be parsable by a LaTeX compiler. All produced content should be enclosed between the \n```latex\n ... \n``` blocks. Do not create document classes or other LaTeX meta commands. Always assume that the document class is already defined. Only produce exactly one latex block with all your content.
Only use either `section`, `subsection`, paragraph`, `texttt`, `textbf`, `emph` or `citep` commands to structure your content. Do not use any other LaTeX commands.
The following is an example of your expected output:

[Example]
```latex
\\documentclass{{article}}
\\begin{{document}}
% TODO: your content here
\\end{{document}}
```

{description}
"""


class Paper(Function):
    def __init__(self, *sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = Sequence(*sequence)

    def forward(self, task, **kwargs):
        # execute the sequence of tasks
        self.sequence(task, **kwargs)
        # access results from the global root node metadata
        root_res    = self.root.metadata._expr_results
        # return the reversed results
        reverse_res = str(list(reversed(root_res)))
        # create
        return super().forward(task | reverse_res, **kwargs)

    @property
    def static_context(self):
        return PAPER_STATIC_CONTEXT.format(description='The paper must include a title, abstract, introduction and related work and method sections.')


class Context(Conversation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_print   = False
        self.prompt       = 'Replace the % TODO: with your content and follow the task description below.'

    def forward(self, task, *args, **kwargs):
        function = Function(self.prompt,
                            post_processors=[StripPostProcessor(), CodeExtractPostProcessor()],
                            static_context=self.static_context,
                            dynamic_context=self.dynamic_context)
        return function(task, *args, **kwargs)

    @property
    def description(self):
        raise NotImplementedError()

    @property
    def static_context(self):
        return PAPER_STATIC_CONTEXT.format(description=self.description)


class Source(Context):
    @property
    def description(self):
        return """[Task]
Summarize the SymbolicAI framework to use it as a conditioning context for a large Language model like GPT-3.
Do not create any sections or subsections. Only write a coherent text about the framework.
"""

class Method(Context):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self.source = source

    def forward(self, task, **kwargs):
        summary = self.source(task, **kwargs)
        # update the dynamic context globally for all types
        self.adapt(context=summary, types=[RelatedWork, Abstract, Title, Introduction, Cite])
        return super().forward(task | summary, **kwargs)

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
    def __init__(self, *citations, **kwargs):
        super().__init__(**kwargs)
        self.citations = Parallel(*citations, sequential=True)

    def forward(self, task, **kwargs):
        # execute the parallel tasks
        res = self.citations(task, **kwargs)
        return super().forward(res, **kwargs)

    @property
    def description(self):
        return """[Task]
Write a coherent related work section in the context of the paper and based on the provided citation sources.
"""


class Introduction(Context):
    def __init__(self, *citations, **kwargs):
        super().__init__(**kwargs)
        self.citations = Parallel(*citations, sequential=True)

    def forward(self, task, **kwargs):
        # execute the parallel tasks
        res = self.citations(task, **kwargs)
        return super().forward(res, **kwargs)

    @property
    def description(self):
        return """[Task]
Write a coherent related work section in the context of the paper and based on the provided citation sources.
"""


class Abstract(Context):
    @property
    def description(self):
        return """[Task]
Write an abstract for the paper.
"""


class Title(Context):
    @property
    def description(self):
        return """[Task]
Write a title for the paper.
"""
