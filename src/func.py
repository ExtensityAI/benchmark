from symai import Expression, Function


FUNCTION_DESCRIPTION = '''
# TODO: Your function description here.
{template}
'''


class MyExpression(Expression):
    def __init__(self):
        super().__init__()
        self.fn = Function(FUNCTION_DESCRIPTION)

    def forward(self, data, template: str = '', *args, **kwargs):
        data = self._to_symbol(data)
        self.fn.format(template=template)
        return self.fn(data, *args, **kwargs)