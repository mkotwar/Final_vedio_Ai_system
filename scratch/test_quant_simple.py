import time
import torch
import gc
import traceback

def run_test(use_4bit: bool):
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        print("Import error:", traceback.format_exc())
        return

    print(f"\n{'='*40}")
    print(f"TESTING 4-BIT QUANTIZATION: {use_4bit}")
    print(f"{'='*40}")
    
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=512*28*28)

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
            device_map="auto",
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
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail and list all objects and events."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
    
    # Real Generation Test
    print("Running Actual Generation Test (100 tokens)...")
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    t1 = time.perf_counter()
    
    gen_time = t1 - t0
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    tokens_generated = len(generated_ids_trimmed[0])
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"\n--- RESULTS ({'4-Bit' if use_4bit else 'BF16'}) ---")
    print(f"Generation Time: {gen_time:.2f} sec")
    print(f"Tokens Per Second: {tps:.2f} t/s")
    
    del model
    del processor
    del inputs
    del generated_ids
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_test(use_4bit=False)
    run_test(use_4bit=True)
