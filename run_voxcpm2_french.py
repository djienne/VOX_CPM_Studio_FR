import argparse
import gc
import os
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
from voxcpm.model import voxcpm2 as voxcpm2_mod


MODEL_ID = "openbmb/VoxCPM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="French-first VoxCPM2 runner with VRAM cleanup and an 8 GB friendly loader patch."
    )
    parser.add_argument(
        "--text",
        default="Bonjour, ceci est un test de synthese vocale en francais avec VoxCPM2.",
        help="French text to synthesize.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Optional UTF-8 text file to synthesize. When set, the text is chunked into sentence groups.",
    )
    parser.add_argument(
        "--output",
        default="sortie_fr.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional French reference audio for cloning.",
    )
    parser.add_argument(
        "--reference-start-sec",
        type=float,
        default=0.0,
        help="Start offset in seconds when trimming a reference clip.",
    )
    parser.add_argument(
        "--reference-duration-sec",
        type=float,
        default=None,
        help="If set, create an intermediate clipped reference WAV of this duration.",
    )
    parser.add_argument(
        "--prompt-audio",
        default=None,
        help="Optional prompt audio for continuation mode.",
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Transcript matching --prompt-audio for continuation mode.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=6,
        help="Inference timesteps for a quick test.",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance value.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=512,
        help="Reduced max_length used to shrink KV-cache memory on 8 GB GPUs.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Generation max_len passed to VoxCPM.generate.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--chunk-target-chars",
        type=int,
        default=220,
        help="Target chunk size in characters when using --text-file.",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=320,
        help="Hard chunk size ceiling in characters when using --text-file.",
    )
    parser.add_argument(
        "--chunk-pause-ms",
        type=int,
        default=180,
        help="Pause inserted between file chunks in milliseconds.",
    )
    return parser.parse_args()


def cleanup_vram(note: str) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    print_gpu_memory(note)


def print_gpu_memory(note: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(
            f"[{note}] free={free_bytes / 1024**2:.0f} MiB total={total_bytes / 1024**2:.0f} MiB "
            f"allocated={allocated / 1024**2:.0f} MiB reserved={reserved / 1024**2:.0f} MiB"
        )
    except Exception as exc:
        print(f"[{note}] GPU memory query unavailable: {exc}")


def maybe_make_reference_clip(
    reference_path: str | None,
    start_sec: float,
    duration_sec: float | None,
) -> str | None:
    if reference_path is None:
        return None
    if duration_sec is None:
        return reference_path

    src = Path(reference_path).resolve()
    audio, sample_rate = librosa.load(src, sr=None, mono=True)
    start_sample = max(0, int(start_sec * sample_rate))
    end_sample = min(len(audio), start_sample + int(duration_sec * sample_rate))
    clipped = audio[start_sample:end_sample]
    output = src.with_name(f"{src.stem}_clip_{start_sec:.1f}s_{duration_sec:.1f}s.wav")
    sf.write(output, clipped, sample_rate)
    print(f"created intermediate reference: {output}")
    return str(output)


def load_text(text: str, text_file: str | None) -> str:
    if text_file is None:
        return text
    return Path(text_file).resolve().read_text(encoding="utf-8").strip()


def split_long_sentence(sentence: str, hard_limit: int) -> list[str]:
    parts: list[str] = []
    remaining = sentence.strip()
    while len(remaining) > hard_limit:
        split_at = -1
        for token in [", ", "; ", ": ", " et ", " ou ", " mais "]:
            candidate = remaining.rfind(token, 0, hard_limit)
            if candidate > split_at:
                split_at = candidate + len(token.strip())
        if split_at <= 0:
            split_at = hard_limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_text(text: str, target_chars: int, max_chars: int) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized).strip()
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    sentences: list[str] = []
    for paragraph in paragraphs:
        sentence_parts = re.split(r"(?<=[.!?…])\s+", paragraph)
        for part in sentence_parts:
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            if len(part) <= max_chars:
                sentences.append(part)
            else:
                sentences.extend(split_long_sentence(part, max_chars))

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= target_chars or not current:
            current = candidate
            continue
        chunks.append(current.strip())
        current = sentence
    if current:
        chunks.append(current.strip())
    return chunks


def move_runtime_modules_to_cuda(model) -> None:
    runtime_modules = [
        model.base_lm,
        model.residual_lm,
        model.feat_encoder,
        model.feat_decoder,
        model.fsq_layer,
        model.enc_to_lm_proj,
        model.lm_to_dit_proj,
        model.res_to_dit_proj,
        model.fusion_concat_proj,
        model.stop_proj,
        model.stop_head,
    ]
    for module in runtime_modules:
        module.to(model.device)


def keep_audio_vae_on_cpu(model) -> None:
    model.audio_vae = model.audio_vae.to("cpu", dtype=torch.float32)
    original_encode = model.audio_vae.encode
    original_decode = model.audio_vae.decode

    def encode_cpu(audio, sample_rate):
        return original_encode(audio.to("cpu"), sample_rate)

    def decode_cpu(latent):
        return original_decode(latent.to("cpu", dtype=torch.float32))

    model.audio_vae.encode = encode_cpu
    model.audio_vae.decode = decode_cpu


def patch_voxcpm2_loader(context_limit: int) -> None:
    def patched_from_local(cls, path: str, optimize: bool = True, training: bool = False, lora_config=None):
        config_path = os.path.join(path, "config.json")
        config = voxcpm2_mod.VoxCPMConfig.model_validate_json(Path(config_path).read_text(encoding="utf-8"))
        original_max_length = config.max_length
        config.max_length = min(config.max_length, context_limit)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"patched config: device={config.device}, dtype={config.dtype}, "
            f"max_length={config.max_length} (was {original_max_length})"
        )

        tokenizer = voxcpm2_mod.LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = voxcpm2_mod.AudioVAEV2(config=audio_vae_config) if audio_vae_config else voxcpm2_mod.AudioVAEV2()

        audiovae_safetensors_path = os.path.join(path, "audiovae.safetensors")
        audiovae_pth_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(audiovae_safetensors_path) and voxcpm2_mod.SAFETENSORS_AVAILABLE:
            print(f"loading AudioVAE from: {audiovae_safetensors_path}")
            vae_state_dict = voxcpm2_mod.load_file(audiovae_safetensors_path, device="cpu")
        elif os.path.exists(audiovae_pth_path):
            print(f"loading AudioVAE from: {audiovae_pth_path}")
            checkpoint = torch.load(audiovae_pth_path, map_location="cpu", weights_only=True)
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError("AudioVAE checkpoint not found in local model directory.")

        model = cls(config, tokenizer, audio_vae, lora_config)
        if not training:
            model = model.to(voxcpm2_mod.get_dtype(model.config.dtype))

        model.audio_vae = model.audio_vae.to(torch.float32)

        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(safetensors_path) and voxcpm2_mod.SAFETENSORS_AVAILABLE:
            print(f"loading model weights from: {safetensors_path}")
            model_state_dict = voxcpm2_mod.load_file(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_model_path):
            print(f"loading model weights from: {pytorch_model_path}")
            checkpoint = torch.load(pytorch_model_path, map_location="cpu", weights_only=True)
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError("Main model checkpoint not found in local model directory.")

        for key, value in vae_state_dict.items():
            model_state_dict[f"audio_vae.{key}"] = value

        model.load_state_dict(model_state_dict, strict=False)

        if training:
            return model

        if model.device == "cuda":
            move_runtime_modules_to_cuda(model)
            keep_audio_vae_on_cpu(model)
        else:
            model = model.to(model.device)

        return model.eval().optimize(disable=not optimize)

    voxcpm2_mod.VoxCPM2Model.from_local = classmethod(patched_from_local)


def validate_args(args: argparse.Namespace) -> None:
    if (args.prompt_audio is None) != (args.prompt_text is None):
        raise ValueError("--prompt-audio and --prompt-text must be provided together.")
    if args.reference_duration_sec is not None and args.reference_duration_sec <= 0:
        raise ValueError("--reference-duration-sec must be positive.")
    if args.context_limit <= 0:
        raise ValueError("--context-limit must be positive.")
    if args.max_len <= 0:
        raise ValueError("--max-len must be positive.")
    if args.text_file is not None and not Path(args.text_file).exists():
        raise FileNotFoundError(f"Text file not found: {args.text_file}")
    if args.chunk_target_chars <= 0:
        raise ValueError("--chunk-target-chars must be positive.")
    if args.chunk_max_chars < args.chunk_target_chars:
        raise ValueError("--chunk-max-chars must be >= --chunk-target-chars.")
    if args.chunk_pause_ms < 0:
        raise ValueError("--chunk-pause-ms must be >= 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu name: {torch.cuda.get_device_name(0)}")

    cleanup_vram("before-load")
    patch_voxcpm2_loader(args.context_limit)

    model_path = snapshot_download(repo_id=MODEL_ID, cache_dir=args.cache_dir, local_files_only=False)
    print(f"local model path: {model_path}")

    prepared_reference = maybe_make_reference_clip(
        args.reference,
        args.reference_start_sec,
        args.reference_duration_sec,
    )
    text_to_speak = load_text(args.text, args.text_file)
    text_chunks = [text_to_speak]
    if args.text_file is not None:
        text_chunks = chunk_text(text_to_speak, args.chunk_target_chars, args.chunk_max_chars)
        print(f"loaded text file: {Path(args.text_file).resolve()}")
        print(f"chunk count: {len(text_chunks)}")

    model = None
    try:
        model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False,
            optimize=False,
        )
        print_gpu_memory("after-load")
        chunk_wavs: list[np.ndarray] = []
        pause_samples = int(model.tts_model.sample_rate * (args.chunk_pause_ms / 1000.0))
        pause_audio = np.zeros(pause_samples, dtype=np.float32) if pause_samples > 0 else None

        for index, chunk_text_value in enumerate(text_chunks, start=1):
            print(f"generating chunk {index}/{len(text_chunks)} ({len(chunk_text_value)} chars)")
            generate_kwargs = dict(
                text=chunk_text_value,
                cfg_value=args.cfg_value,
                inference_timesteps=args.timesteps,
                max_len=args.max_len,
            )
            if prepared_reference is not None:
                generate_kwargs["reference_wav_path"] = prepared_reference
            if args.prompt_audio is not None:
                generate_kwargs["prompt_wav_path"] = args.prompt_audio
                generate_kwargs["prompt_text"] = args.prompt_text

            with torch.inference_mode():
                wav = model.generate(**generate_kwargs)
            chunk_wavs.append(np.asarray(wav, dtype=np.float32))
            if pause_audio is not None and index < len(text_chunks):
                chunk_wavs.append(pause_audio.copy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_gpu_memory(f"after-chunk-{index}")

        wav = np.concatenate(chunk_wavs) if len(chunk_wavs) > 1 else chunk_wavs[0]

        output_path = Path(args.output).resolve()
        sf.write(output_path, wav, model.tts_model.sample_rate)
        print(f"saved: {output_path}")
        print_gpu_memory("after-generate")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA OOM while loading or generating.")
            print("Try closing GPU-heavy apps, lowering --context-limit, or trimming the reference clip.")
        raise
    finally:
        if model is not None:
            del model
        cleanup_vram("after-cleanup")


if __name__ == "__main__":
    main()
