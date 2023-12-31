import logging
import google.generativeai as genai

from typing import List, Optional

from symai.backend.base import Engine

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GoogleGeminiEngine(Engine):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        logger              = logging.getLogger('vertexai')
        logger.setLevel(logging.WARNING)
        # Initialize the Vertex AI project
        self.api_key        = api_key
        genai.configure(api_key=api_key)
        # Create a generative model instance from Vertex AI
        self.model = genai.GenerativeModel(model_name=model)
        self.max_tokens     = 32_760 - 100 # TODO: account for tolerance. figure out how their magic number works to compute reliably the precise max token size
        self.seed           = None
        self.except_remedy  = None

    def id(self) -> str:
        if   self.config['NEUROSYMBOLIC_ENGINE_MODEL'] and \
             self.config['NEUROSYMBOLIC_ENGINE_MODEL'].startswith('gemini'):
            return 'neurosymbolic'
        return super().id() # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'NEUROSYMBOLIC_ENGINE_MODEL' in kwargs:
            self.model     = kwargs['NEUROSYMBOLIC_ENGINE_MODEL']
        if 'seed' in kwargs:
            self.seed      = kwargs['seed']
        if 'except_remedy' in kwargs:
            self.except_remedy = kwargs['except_remedy']

    def compute_remaining_tokens(self, prompts: list) -> int:
        return int((8_192) * 0.99) # TODO: figure out how their magic number works to compute reliably the precise max token size

    def forward(self, argument):
        kwargs              = argument.kwargs
        prompts_            = argument.prop.prepared_input

        # send prompt to GPT-X Chat-based
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        model               = kwargs['model'] if 'model' in kwargs else self.model
        seed                = kwargs['seed'] if 'seed' in kwargs else self.seed

        # convert map to list of strings
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts_)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.1
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 1
        top_k               = kwargs['top_k'] if 'top_k' in kwargs else 40
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else self.except_remedy

        try:
            res = model.generate_content(
                prompts_,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                }
            )

        except Exception as e:
            callback = model.generate_content
            kwargs['model'] = kwargs['model'] if 'model' in kwargs else self.model
            if except_remedy is not None:
                res = except_remedy(self, e, callback, argument)
            else:
                raise e

        metadata = {}
        output   = [res.text]
        return output, metadata

    def prepare(self, argument):
        if argument.prop.raw_input:
            if not argument.prop.processed_input:
                raise ValueError('Need to provide a prompt instruction to the engine if raw_input is enabled.')
            argument.prop.prepared_input = argument.prop.processed_input
            return

        _non_verbose_output = """[META INSTRUCTIONS START]\nYou do not output anything else, like verbose preambles or post explanation, such as "Sure, let me...", "Hope that was helpful...", "Yes, I can help you with that...", etc. Consider well formatted output, e.g. for sentences use punctuation, spaces etc. or for code use indentation, etc. Never add meta instructions information to your output!\n"""
        user:   str = ""
        system: str = ""

        if argument.prop.disable_verbose_output_suppression:
            system += _non_verbose_output
        system = f'{system}\n' if system and len(system) > 0 else ''

        ref = argument.prop.instance
        static_ctxt, dyn_ctxt = ref.global_context
        if len(static_ctxt) > 0:
            system += f"[STATIC CONTEXT]\n{static_ctxt}\n\n"

        if len(dyn_ctxt) > 0:
            system += f"[DYNAMIC CONTEXT]\n{dyn_ctxt}\n\n"

        payload = argument.prop.payload
        if argument.prop.payload:
            system += f"[ADDITIONAL CONTEXT]\n{str(payload)}\n\n"

        examples: List[str] = argument.prop.examples
        if examples and len(examples) > 0:
            system += f"[EXAMPLES]\n{str(examples)}\n\n"

        if argument.prop.prompt is not None and len(argument.prop.prompt) > 0:
            val = str(argument.prop.prompt)
            system += f"[INSTRUCTION]\n{val}"

        suffix: str = str(argument.prop.processed_input)

        if '[SYSTEM_INSTRUCTION::]: <<<' in suffix and argument.prop.parse_system_instructions:
            parts = suffix.split('\n>>>\n')
            # first parts are the system instructions
            c = 0
            for i, p in enumerate(parts):
                if 'SYSTEM_INSTRUCTION' in p:
                    system += f"{p}\n"
                    c += 1
                else:
                    break
            # last part is the user input
            suffix = '\n>>>\n'.join(parts[c:])
        user += f"{suffix}"

        if argument.prop.template_suffix:
            user += f"\n[[PLACEHOLDER]]\n{str(argument.prop.template_suffix)}\n\n"
            user += f"Only generate content for the placeholder `[[PLACEHOLDER]]` following the instructions and context information. Do NOT write `[[PLACEHOLDER]]` or anything else in your output.\n\n"

        argument.prop.prepared_input = system + '\n' + user
