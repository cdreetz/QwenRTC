import os
import torch

from transformers import Qwen2_5OmniProcessor
from vllm import LLM, SamplingParams
from qwen_omni_utils import process_mm_info

# vLLM engine v1 not supported yet
os.environ['VLLM_USE_V1'] = '0'

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

llm = LLM(
    model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.9,
    tensor_parallel_size=torch.cuda.device_count(),
    limit_mm_per_prompt={'image': 1, 'video': 1, 'audio': 1},
    seed=1234,
)

sampling_params = SamplingParams(
    temperature=1e-6,
    max_tokens=512,
)

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

inputs = {
    'prompt': text[0],
    'multi_modal_data': {},
    "mm_processor_kwargs": {
        "use_audio_in_video": True,
    },
}


if images is not None:
    inputs['multi_modal_data']['image'] = images
if videos is not None:
    inputs['multi_modal_data']['video'] = videos
if audios is not None:
    inputs['multi_modal_data']['audio'] = audios

outputs = llm.generate(inputs, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
