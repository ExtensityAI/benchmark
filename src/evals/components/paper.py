from symai import Function
from symai.components import Sequence, Parallel
from symai.extended import Conversation
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor


SYMBOLIC_AI_PAPER = """Write a scientific paper about the machine learning framework called SymbolicAI which operates on the following principles:
- Symbolic methods
- Sub-symbolic methods
- Neural-symbolic methods
- Probabilistic programming methods
- Cognitive architectures
Be precise in your writing and follow a scientific style. Do not use any colloquial language. However, formulate simple and understandable sentences."""


PAPER_STATIC_CONTEXT = """[General Context]
{context}

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
    def __init__(self, *sequence, context: str = SYMBOLIC_AI_PAPER, **kwargs):
        super().__init__(**kwargs)
        self.sequence = Sequence(*sequence)
        self.context  = context

    def forward(self, task, **kwargs):
        # execute the sequence of tasks
        res         = self.sequence(task, **kwargs)
        # access results from the global root node metadata
        results     = self.linker.results
        # return the reversed results
        reverse_res = str(list(reversed(list(results.values()))))
        # create the final task by concatenating the results
        return super().forward(task | reverse_res | res, **kwargs)

    @property
    def static_context(self):
        return PAPER_STATIC_CONTEXT.format(context=self.context, description='The final paper must include the title an abstract and a related work section and method section.')


class Context(Conversation):
    def __init__(self, context: str = SYMBOLIC_AI_PAPER, **kwargs):
        super().__init__(**kwargs)
        self.auto_print   = False
        self.prompt       = 'Replace the % TODO: with your content and follow the task description below.'
        self.context      = context

    def forward(self, task, *args, **kwargs):
        function = Function(self.prompt,
                            post_processors=[StripPostProcessor(), CodeExtractPostProcessor()],
                            static_context=self.static_context,
                            dynamic_context=self.dynamic_context)
        return function(f"{task}\n[Source]\n{self.history()}", *args, **kwargs)

    @property
    def description(self):
        raise NotImplementedError()

    @property
    def static_context(self):
        return PAPER_STATIC_CONTEXT.format(context=self.context, description=self.description)


class Source(Context):
    @property
    def description(self):
        return """[Task]
Summarize the referenced method to use it as a conditioning context for a large Language model like GPT-3.
Do not create any sections or subsections. Only write one coherent text about the main principles and concepts of the method.
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
Your goal is to write the method section which describes the main approach and principles used. Add one methodology section with one consistent paragraph. Provide citations and references.
"""


class Cite(Source):
    @property
    def description(self):
        return """[Task]
Write a short two sentence related work summary in the context of the paper. Do not add any sections or subsections.
"""


class RelatedWork(Context):
    def __init__(self, *citations, **kwargs):
        super().__init__(**kwargs)
        self.citations = Parallel(*citations, sequential=True) # to avoid API rate limits process parallel citations sequentially

    def forward(self, task, **kwargs):
        # execute the parallel tasks
        res = self.citations(task, **kwargs)
        return super().forward(res, **kwargs)

    @property
    def description(self):
        return """[Task]
Write a coherent related work section in the context of the paper and based on the provided citation sources. Add one related work section with one consistent paragraph. Provide citations and references.
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
Write a coherent introduction section in the context of the paper and based on the provided context. Add one introduction section with one consistent paragraph. Provide citations and references.
"""


class Abstract(Context):
    @property
    def description(self):
        return """[Task]
Write the paper abstract given the provided context. Add one abstract section with one consistent paragraph.
"""


class Title(Context):
    @property
    def description(self):
        return """[Task]
Write the paper title given the provided context. Add one title tag for the document.
"""
