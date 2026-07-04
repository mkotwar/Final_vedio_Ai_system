from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SMOL_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOL_REQUIRED_PACKAGES = ("num2words",)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _debug_root() -> Path:
    return Path(__file__).resolve().parent / "debug_runs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except Exception:
        return str(path)


def _read_prompt(args: argparse.Namespace) -> str:
    prompt = str(args.prompt or "").strip()
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Provide --prompt or --prompt-file.")
    return prompt


def _resolve_images(args: argparse.Namespace) -> list[Path]:
    image_values = list(args.image or [])
    if args.images_dir:
        image_dir = Path(args.images_dir).expanduser()
        if not image_dir.is_absolute():
            image_dir = (_repo_root() / image_dir).resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_values.extend(str(path) for path in sorted(image_dir.glob(suffix)))

    resolved: list[Path] = []
    for raw_path in image_values:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        resolved.append(path)
    if not resolved:
        raise ValueError("Provide at least one --image or --images-dir.")
    return resolved


class QwenImageAdapter:
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.device.startswith("cuda"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)

    def generate_one(self, image_path: Path, prompt: str) -> dict[str, Any]:
        self.load()
        assert self.processor is not None
        assert self.model is not None

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        start_time = time.perf_counter()
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([message])
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        elapsed = time.perf_counter() - start_time
        return {
            "output_text": output_text,
            "elapsed_seconds": round(elapsed, 3),
            "output_length_chars": len(output_text),
        }


class SmolImageAdapter:
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.device.startswith("cuda"):
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id)
            self.model.to(self.device)

    def generate_one(self, image_path: Path, prompt: str) -> dict[str, Any]:
        self.load()
        assert self.processor is not None
        assert self.model is not None

        image = Image.open(image_path).convert("RGB")
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        start_time = time.perf_counter()
        model_inputs = self.processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        prompt_length = model_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        elapsed = time.perf_counter() - start_time
        return {
            "output_text": generated_text,
            "elapsed_seconds": round(elapsed, 3),
            "output_length_chars": len(generated_text),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _check_importable_packages(package_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for package_name in package_names:
        try:
            importlib.import_module(package_name)
        except Exception:
            missing.append(package_name)
    return missing


def _build_error_result(error_message: str) -> dict[str, Any]:
    return {
        "output_text": "",
        "elapsed_seconds": None,
        "output_length_chars": 0,
        "status": "error",
        "error": error_message,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# VLM Model Comparison",
        "",
        f"- Prompt: `{payload['prompt']}`",
        f"- Image count: `{summary['image_count']}`",
        f"- Device: `{summary['device']}`",
        f"- Qwen avg seconds: `{summary['qwen_average_seconds']}`",
        f"- SmolVLM avg seconds: `{summary['smol_average_seconds']}`",
        f"- Faster model: `{summary['faster_model']}`",
        "",
        "## Per Image",
        "",
    ]
    for item in payload["items"]:
        qwen_time = item["qwen"]["elapsed_seconds"]
        smol_time = item["smolvlm"]["elapsed_seconds"]
        lines.extend(
            [
                f"### {item['image_name']}",
                "",
                f"- Qwen status: `{item['qwen'].get('status', 'ok')}`",
                f"- Qwen time: `{qwen_time if qwen_time is not None else 'n/a'}s`",
                f"- SmolVLM status: `{item['smolvlm'].get('status', 'ok')}`",
                f"- SmolVLM time: `{smol_time if smol_time is not None else 'n/a'}s`",
                "",
                "**Qwen output**",
                "",
                item["qwen"]["output_text"] or "_empty_",
                "",
            ]
        )
        if item["qwen"].get("error"):
            lines.extend(["Qwen error:", "", item["qwen"]["error"], ""])
        lines.extend(
            [
                "**SmolVLM output**",
                "",
                item["smolvlm"]["output_text"] or "_empty_",
                "",
            ]
        )
        if item["smolvlm"].get("error"):
            lines.extend(["SmolVLM error:", "", item["smolvlm"]["error"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen and SmolVLM on the same image prompt set.")
    parser.add_argument("--image", action="append", default=[], help="Image path. Repeat for multiple images.")
    parser.add_argument("--images-dir", default="", help="Directory of images to compare.")
    parser.add_argument("--prompt", default="", help="Prompt string.")
    parser.add_argument("--prompt-file", default="", help="Path to prompt text file.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    parser.add_argument("--qwen-model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--smol-model-id", default=SMOL_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    prompt = _read_prompt(args)
    image_paths = _resolve_images(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_repo_root() / output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _debug_root() / f"vlm_compare_{timestamp}"
    _ensure_dir(output_dir)

    qwen = QwenImageAdapter(args.qwen_model_id, device, args.max_new_tokens)
    smol = SmolImageAdapter(args.smol_model_id, device, args.max_new_tokens)
    smol_missing_packages = _check_importable_packages(SMOL_REQUIRED_PACKAGES)

    input_payload = {
        "prompt": prompt,
        "device": device,
        "qwen_model_id": args.qwen_model_id,
        "smol_model_id": args.smol_model_id,
        "max_new_tokens": args.max_new_tokens,
        "smol_required_packages": list(SMOL_REQUIRED_PACKAGES),
        "smol_missing_packages": smol_missing_packages,
        "images": [{"path": str(path), "repo_relative_path": _to_repo_relative(path), "name": path.name} for path in image_paths],
    }
    _write_json(output_dir / "01_inputs.json", input_payload)

    qwen_load_error = ""
    smol_load_error = ""
    try:
        print("[vlm-compare] Loading Qwen model...")
        qwen.load()
    except Exception as exc:
        qwen_load_error = str(exc).strip() or repr(exc)
        print(f"[vlm-compare] Qwen load failed: {qwen_load_error}", file=sys.stderr)

    if smol_missing_packages:
        smol_load_error = (
            "Missing required package(s) for SmolVLM: "
            + ", ".join(smol_missing_packages)
            + ". Install with: pip install "
            + " ".join(smol_missing_packages)
        )
        print(f"[vlm-compare] {smol_load_error}", file=sys.stderr)
    else:
        try:
            print("[vlm-compare] Loading SmolVLM model...")
            smol.load()
        except Exception as exc:
            smol_load_error = str(exc).strip() or repr(exc)
            print(f"[vlm-compare] SmolVLM load failed: {smol_load_error}", file=sys.stderr)

    items: list[dict[str, Any]] = []
    qwen_times: list[float] = []
    smol_times: list[float] = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[vlm-compare] Processing image {index}/{len(image_paths)}: {image_path.name}")
        if qwen_load_error:
            qwen_result = _build_error_result(qwen_load_error)
        else:
            try:
                qwen_result = qwen.generate_one(image_path, prompt)
                qwen_result["status"] = "ok"
                qwen_times.append(float(qwen_result["elapsed_seconds"]))
            except Exception as exc:
                qwen_result = _build_error_result(str(exc).strip() or repr(exc))

        if smol_load_error:
            smol_result = _build_error_result(smol_load_error)
        else:
            try:
                smol_result = smol.generate_one(image_path, prompt)
                smol_result["status"] = "ok"
                smol_times.append(float(smol_result["elapsed_seconds"]))
            except Exception as exc:
                smol_result = _build_error_result(str(exc).strip() or repr(exc))

        items.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "repo_relative_path": _to_repo_relative(image_path),
                "qwen": qwen_result,
                "smolvlm": smol_result,
            }
        )

    qwen_avg = round(sum(qwen_times) / len(qwen_times), 3) if qwen_times else 0.0
    smol_avg = round(sum(smol_times) / len(smol_times), 3) if smol_times else 0.0
    summary = {
        "image_count": len(items),
        "device": device,
        "qwen_status": "error" if qwen_load_error else "ok",
        "smolvlm_status": "error" if smol_load_error else "ok",
        "qwen_load_error": qwen_load_error or None,
        "smolvlm_load_error": smol_load_error or None,
        "qwen_average_seconds": qwen_avg,
        "smol_average_seconds": smol_avg,
        "faster_model": "qwen" if qwen_avg < smol_avg else "smolvlm" if smol_avg < qwen_avg else "tie",
    }
    output_payload = {
        "prompt": prompt,
        "summary": summary,
        "items": items,
    }
    _write_json(output_dir / "02_comparison_results.json", output_payload)
    _write_markdown(output_dir / "03_comparison_report.md", output_payload)

    print(f"[vlm-compare] Output dir: {output_dir}")
    print(f"[vlm-compare] Qwen avg: {qwen_avg}s")
    print(f"[vlm-compare] SmolVLM avg: {smol_avg}s")


if __name__ == "__main__":
    main()
