import sys
import time
import torch
import gc
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class LoggerWriter:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

LoggerWriter("c:\\Mukul K\\vinfo1\\video-search-engine\\scratch\\test_quant.out")

def print_vram(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{prefix} VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def run_test(use_4bit: bool):
    print(f"\n{'='*40}")
    print(f"TESTING 4-BIT QUANTIZATION: {use_4bit}")
    print(f"{'='*40}")
    
    # Clean VRAM before test
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("Baseline")

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    processor = AutoProcessor.from_pretrained(
        model_id, 
        min_pixels=256*28*28, 
        max_pixels=512*28*28
    )

    t0 = time.perf_counter()
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"":"cuda:0"},
            attn_implementation="sdpa",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        
    t1 = time.perf_counter()
    print(f"Model Load Time: {t1 - t0:.2f}s")
    print_vram("After Model Load")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Describe this image in detail and list all objects and events."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    print_vram("After Processor")
    
    # Warmup
    print("Running Warmup Generation (20 tokens)...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate(**inputs, max_new_tokens=20)
        
    print_vram("After Warmup")
    
    # Real Generation Test
    print("Running Actual Generation Test (200 tokens)...")
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    t1 = time.perf_counter()
    
    print_vram("After Generation")
    
    gen_time = t1 - t0
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    tokens_generated = len(generated_ids_trimmed[0])
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"\n--- RESULTS ({'4-Bit' if use_4bit else 'BF16'}) ---")
    print(f"Generation Time: {gen_time:.2f} sec")
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Tokens Per Second: {tps:.2f} t/s")
    print(f"Output Length: {len(out_text)} chars")
    
    del model
    del processor
    del inputs
    del generated_ids
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        run_test(use_4bit=False)
        run_test(use_4bit=True)
    except Exception as e:
        print(f"Error during test: {e}")
