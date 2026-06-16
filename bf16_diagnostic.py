"""
BFloat16 Fix Validation — Quick test
Loads the model with the fixed code path and runs a single image inference.

Usage: .\.venv\Scripts\python.exe bf16_diagnostic.py
"""
import os, sys, time
sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
os.chdir(r"c:\Mukul K\vinfo1\video-search-engine")

def main():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print()

    from app.services.qwen_vlm_hf import NativeQwenTransformersService
    
    print("--- Loading model ---")
    t0 = time.time()
    NativeQwenTransformersService.load_model()
    print(f"Loaded in {time.time()-t0:.1f}s")
    print()
    
    model = NativeQwenTransformersService._model
    print(f"model.dtype = {model.dtype}")
    print(f"First param dtype = {next(model.parameters()).dtype}")
    print(f"First param device = {next(model.parameters()).device}")
    print(f"Has hf_device_map = {hasattr(model, 'hf_device_map')}")
    if hasattr(model, 'hf_device_map'):
        print(f"  device_map = {model.hf_device_map}")
    print()
    
    # Check q_proj specifically (the crashing layer)
    try:
        qp = next(model.model.language_model.layers[0].self_attn.q_proj.parameters())
        print(f"q_proj weight dtype: {qp.dtype}")
        print(f"q_proj weight device: {qp.device}")
    except Exception as e:
        print(f"q_proj probe failed: {e}")
    print()
    
    # Find a test frame
    from pathlib import Path
    test_image = None
    frames_dir = Path(r"c:\Mukul K\vinfo1\video-search-engine\data\frames")
    for d in frames_dir.iterdir():
        if d.is_dir():
            jpgs = list(d.glob("*.jpg"))
            if jpgs:
                test_image = jpgs[0]
                break
    
    if not test_image:
        print("No test image found!")
        return
        
    print(f"Test image: {test_image}")
    print()
    
    # Run inference
    print("--- Running generate_batch (single image) ---")
    try:
        t0 = time.time()
        results = NativeQwenTransformersService.generate_batch(
            [test_image], "Describe this image briefly."
        )
        elapsed = time.time() - t0
        print(f"SUCCESS in {elapsed:.1f}s")
        print(f"Output: {results[0][:300]}")
        print()
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print()
        print("=== VERDICT: BF16 FIX CONFIRMED ===")
    except RuntimeError as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=== VERDICT: BF16 FIX DID NOT WORK ===")

if __name__ == "__main__":
    main()
