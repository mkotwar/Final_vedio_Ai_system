@echo off
call .venv\Scripts\activate
python get_hardware2.py
python benchmark_ollama.py > benchmark_ollama_out.txt
python benchmark_hf.py > benchmark_hf_out.txt
python benchmark_vllm.py > benchmark_vllm_out.txt
