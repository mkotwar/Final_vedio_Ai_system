import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("TENDER_DEMO_QWEN_API_KEY")
base_url = os.getenv("TENDER_DEMO_QWEN_BASE_URL", "https://openrouter.ai/api/v1")
model = os.getenv("TENDER_DEMO_QWEN_MODEL", "qwen/qwen3-vl-8b-instruct")

if not api_key:
    raise RuntimeError("TENDER_DEMO_QWEN_API_KEY missing")

print("base_url:", base_url)
print("model:", model)
print("key:", api_key[:8] + "..." + api_key[-4:])

client = OpenAI(
    api_key=api_key.strip(),
    base_url=base_url.strip(),
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Tender Demo VLM Test",
    },
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Reply only with: OpenRouter Qwen working"
        }
    ],
    max_tokens=50,
)

print("TEXT OUTPUT:", response.choices[0].message.content)

# Optional image test. Replace with any real jpg from your debug run.
image_path = r"PUT_IMAGE_PATH_HERE"

if image_path != "PUT_IMAGE_PATH_HERE":
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(p)

    image_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this CCTV/traffic image in one sentence."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                ],
            }
        ],
        max_tokens=150,
        temperature=0,
    )

    print("IMAGE OUTPUT:", response.choices[0].message.content)