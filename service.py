import os
import uuid
from argparse import Namespace
from typing import AsyncGenerator, Optional

import PIL
import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

import fastapi
openai_api_app = fastapi.FastAPI()

MAX_MODEL_LEN = 8192
MAX_IMAGE_SIZE = 640

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-90B-Vision-Instruct")

TOOL_CALL_PARSER = os.getenv("TOOL_CALL_PARSER", "llama3_json")
ENABLE_TOOL_CALL_PARSER = os.getenv("ENABLE_TOOL_CALL_PARSER", True)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

TIMEOUT = int(os.getenv("TIMEOUT", 300))
CONCURRENCY = int(os.getenv("CONCURRENCY", 256))

GPU_COUNT = int(os.getenv("GPU_COUNT", 1))
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", True)
KV_CACHE_TYPE = os.getenv("KV_CACHE_TYPE", "fp8")


def resize(image: PIL.Image.Image, max_size: int = MAX_IMAGE_SIZE):
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        width = int(image.width * ratio)
        height = int(image.height * ratio)
        image = image.resize((width, height))

    return image


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentovllm-llama-3.2-90B-Vision-Instruct",
    traffic={
        "timeout": TIMEOUT,
        "concurrency": CONCURRENCY
    },
)
class VLLM:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import vllm.entrypoints.openai.api_server as vllm_api_server

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            tensor_parallel_size=GPU_COUNT,
            enforce_eager=True,
            max_num_seqs=16,
            limit_mm_per_prompt=dict(image=1)
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        OPENAI_ENDPOINTS = [
            ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
            ["/completions", vllm_api_server.create_completion, ["POST"]],
            ["/models", vllm_api_server.show_available_models, ["GET"]],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint,
                methods=methods,
            )

        model_config = self.engine.engine.get_model_config()
        args = Namespace()
        args.model = MODEL_ID
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.response_role = "assistant"
        args.served_model_name = None
        args.chat_template = None
        args.lora_modules = None
        args.prompt_adapters = None
        args.request_logger = None
        args.disable_log_stats = True
        args.return_tokens_as_token_ids = False
        args.enable_tool_call_parser = True
        args.enable_auto_tool_choice = True
        args.tool_call_parser = "llama3_json"
        args.enable_prompt_tokens_details = False

        vllm_api_server.init_app_state(
            self.engine, model_config, openai_api_app.state, args
        )

    @bentoml.api
    async def generate(
            self,
            image: PIL.Image.Image,
            prompt: str = "Describe this image",
            system_prompt: Optional[str] = SYSTEM_PROMPT,
            max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=self.stop_token_ids,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        engine_inputs = await self.create_image_inputs(
            dict(prompt=prompt, system_prompt=system_prompt, image=resize(image))
        )
        stream = await self.engine.add_request(
            uuid.uuid4().hex, engine_inputs, SAMPLING_PARAM
        )

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)

    async def create_image_inputs(self, inputs):
        from vllm import TextPrompt
        from vllm.multimodal import MultiModalDataBuiltins

        return TextPrompt(
            prompt=PROMPT_TEMPLATE.format(
                user_prompt=inputs["prompt"], system_prompt=inputs["system_prompt"]
            ),
            multi_modal_data=MultiModalDataBuiltins(image=inputs["image"]),
        )
