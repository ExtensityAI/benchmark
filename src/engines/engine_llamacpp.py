import logging
import requests
import json

from typing import List
from requests_toolbelt.multipart.encoder import MultipartEncoder

from symai.backend.base import Engine
from symai.backend.settings import SYMAI_CONFIG
from symai.symbol import Result


class LLaMAResult(Result):
    def __init__(self, value=None, *args, **kwargs):
        super().__init__(value, *args, **kwargs)
        self._value = value
        self.error  = None
        self.raw    = value
        self._perse_result()

    def _perse_result(self):
        val        = json.loads(self.value)
        self.value = val
        if 'error' in val:
            self.error = val['error']
        if 'content' in val:
            self.value = val['content']


class LLaMACppClientEngine(Engine):
    def __init__(self, host: str = 'http://localhost', port: int = 8080, uri: str = 'completion', timeout: int = 240):
        super().__init__()
        logger = logging.getLogger('nesy_client')
        logger.setLevel(logging.WARNING)
        self.config         = SYMAI_CONFIG
        self.host           = host
        self.port           = port
        self.uri            = uri
        self.timeout        = timeout
        self.seed           = None
        self.except_remedy  = None

    def id(self) -> str:
        if  self.config['CAPTION_ENGINE_MODEL'] and \
            'llamacpp' in self.config['CAPTION_ENGINE_MODEL']:
            return 'neurosymbolic'
        return super().id() # default to unregistered

    @property
    def max_tokens(self):
        return 2048

    def compute_remaining_tokens(self, prompts: list) -> int:
        return int((1024) * 0.99) # TODO: figure out how their magic number works to compute reliably the precise max token size

    def forward(self, argument):
        prompts             = argument.prop.prepared_input
        kwargs              = argument.kwargs

        model_kwargs = {}

        # convert map to list of strings
        stop                = kwargs['stop'] if 'stop' in kwargs else None
        seed                = kwargs['seed'] if 'seed' in kwargs else self.seed
        max_tokens          = kwargs['max_tokens'] if 'max_tokens' in kwargs else self.compute_remaining_tokens(prompts)
        temperature         = kwargs['temperature'] if 'temperature' in kwargs else 0.7
        top_p               = kwargs['top_p'] if 'top_p' in kwargs else 0.95
        top_k               = kwargs['top_k'] if 'top_k' in kwargs else 40
        except_remedy       = kwargs['except_remedy'] if 'except_remedy' in kwargs else self.except_remedy

        if stop is not None:
            model_kwargs['stop'] = stop
        if seed is not None:
            model_kwargs['seed'] = seed
        if max_tokens is not None:
            model_kwargs['n_predict'] = max_tokens
        if temperature is not None:
            model_kwargs['temperature'] = temperature
        if top_p is not None:
            model_kwargs['top_p'] = top_p
        if top_k is not None:
            model_kwargs['top_k'] = top_k

        # Create multipart/form-data payload
        # Since the LLaMA server expects a JSON payload, we construct JSON data
        payload = {
            'prompt': prompts[0],
            **model_kwargs
        }
        headers = {'Content-Type': 'application/json'}
        api     = f'{self.host}:{self.port}/{self.uri}'
        try:
            rsp = requests.post(api, json=payload, headers=headers, timeout=self.timeout)
            # Verify the success of the response
            rsp.raise_for_status()
            res = rsp.text
        except requests.exceptions.HTTPError as e:
            if except_remedy is None:
                self.logger.error(f"HTTP error occurred: {e}")
                # Here you can add more sophisticated error handling and recovery
                raise e
            # Retry the request or handle it based on the exception remedy provided
            callback = lambda: requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = except_remedy(self, e, callback, argument)
        except requests.exceptions.RequestException as e:
            # Handle non-HTTP exceptions (e.g., network errors, timeout)
            if except_remedy is None:
                self.logger.error(f"Request error occurred: {e}")
                raise e
            # Retry the request or handle it based on the exception remedy provided
            callback = lambda: requests.post(api, data=payload, headers=headers, timeout=self.timeout)
            res = except_remedy(self, e, callback, argument)
        except Exception as e:
            # Handle unforeseen exceptions
            self.logger.error(f"An unexpected error occurred: {e}")
            raise e

        metadata = {}

        try:
            res = LLaMAResult(res)
        except json.JSONDecodeError:
            # Handle a JSON parse error specifically
            self.logger.error(f"JSON parse error: Invalid response {res}")
            raise Exception(f"Invalid response: {res}")

        rsp = [res]
        output = rsp if isinstance(prompts, list) else rsp[0]
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
            # in this engine, instructions are considered as user prompts
            user += f"[INSTRUCTION]\n{val}"

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

        argument.prop.prepared_input = [ system + "\n" + user ]
