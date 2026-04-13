from __future__ import annotations

import atexit
import re
import secrets
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import TypedDict, cast

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM

from run_voxcpm2_french import MODEL_ID, cleanup_vram, patch_voxcpm2_loader


WORKSPACE = Path(__file__).resolve().parent
OUTPUT_ROOT = WORKSPACE / "gui_outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

DEFAULT_CONTEXT_LIMIT = 512
DEFAULT_CFG_VALUE = 2.0
DEFAULT_TIMESTEPS = 6
DEFAULT_MAX_LEN = 256
DEFAULT_TARGET_CHUNK_SECONDS = 14.0
DEFAULT_WORDS_PER_SECOND = 2.5
DEFAULT_SENTENCE_PAUSE_SECONDS = 0.15
DEFAULT_MERGE_SILENCE_SECONDS = 0.12

MODEL_LOCK = threading.RLock()
MODEL_INSTANCE: VoxCPM | None = None
MODEL_SAMPLE_RATE: int | None = None
MODEL_CONTEXT_LIMIT: int | None = None
MODEL_LOADING = False
MODEL_PRELOAD_ERROR: str | None = None
MODEL_PRELOAD_STARTED = False
DOT_SENTINEL = "<DOT>"
FRENCH_ABBREVIATIONS = (
    "M.",
    "Mme.",
    "Mlle.",
    "Dr.",
    "Pr.",
    "St.",
    "Ste.",
    "etc.",
    "cf.",
    "vs.",
    "p.ex.",
    "N.B.",
)


class ChunkState(TypedDict):
    text: str
    estimated_seconds: float
    actual_seconds: float | None
    audio_path: str | None
    approved: bool
    seed: int | None
    generation_count: int
    oversized: bool


class SessionState(TypedDict):
    session_dir: str | None
    source_name: str | None
    source_path: str | None
    reference_path: str | None
    reference_name: str | None
    target_chunk_seconds: float
    words_per_second: float
    sentence_pause_seconds: float
    merge_silence_seconds: float
    context_limit: int
    chunks: list[ChunkState]
    current_index: int
    merged_path: str | None


def workspace_text_files() -> list[str]:
    files = [path.name for path in WORKSPACE.glob("*.txt")]
    return sorted(files)


def workspace_reference_files() -> list[str]:
    audio_suffixes = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = [path.name for path in WORKSPACE.iterdir() if path.is_file() and path.suffix.lower() in audio_suffixes]
    refs = [name for name in files if name.lower().startswith("ref")]
    return sorted(refs)


def default_reference_name() -> str | None:
    refs = workspace_reference_files()
    for preferred in ["ref_long_19s.mp3", "ref_short.mp3", "ref_long.mp3"]:
        if preferred in refs:
            return preferred
    return refs[0] if refs else None


def safe_random_seed() -> int:
    return secrets.randbelow(2_147_483_647)


def read_text_file(path: str) -> str:
    """Read a text file using a small set of practical encodings."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file: {path}")


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def protect_abbreviations(text: str) -> str:
    """Hide periods in common French abbreviations before sentence splitting."""
    protected = text
    for abbreviation in FRENCH_ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", DOT_SENTINEL))
    return protected


def restore_abbreviations(text: str) -> str:
    """Restore hidden abbreviation periods after sentence splitting."""
    return text.replace(DOT_SENTINEL, ".")


def split_into_sentences(text: str) -> list[str]:
    """Split French-first text into sentences while avoiding common abbreviation cuts."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        protected = protect_abbreviations(paragraph)
        parts = re.split(r"(?:(?<=[.!?])|(?<=\.\.\.))\s+", protected)
        for part in parts:
            sentence = re.sub(r"\s+", " ", restore_abbreviations(part)).strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def estimate_seconds(text: str, words_per_second: float, sentence_pause_seconds: float) -> float:
    words = max(1, word_count(text))
    sentences = max(1, len(split_into_sentences(text)))
    speech_seconds = words / words_per_second
    pause_seconds = max(0, sentences - 1) * sentence_pause_seconds
    return round(speech_seconds + pause_seconds, 2)


def build_chunk_state(
    text: str,
    target_seconds: float,
    words_per_second: float,
    sentence_pause_seconds: float,
) -> ChunkState:
    """Create one chunk state record with derived duration metadata."""
    estimated = estimate_seconds(text, words_per_second, sentence_pause_seconds)
    return {
        "text": text,
        "estimated_seconds": estimated,
        "actual_seconds": None,
        "audio_path": None,
        "approved": False,
        "seed": None,
        "generation_count": 0,
        "oversized": estimated > target_seconds,
    }


def chunk_text_for_target_duration(
    text: str,
    target_seconds: float,
    words_per_second: float,
    sentence_pause_seconds: float,
) -> list[ChunkState]:
    """Group whole sentences into chunks near the requested target duration."""
    sentences = split_into_sentences(text)
    chunks: list[ChunkState] = []

    if not sentences:
        return chunks

    current_sentences: list[str] = []
    current_estimate = 0.0

    for sentence in sentences:
        sentence_estimate = estimate_seconds(sentence, words_per_second, sentence_pause_seconds)
        if current_sentences and current_estimate + sentence_estimate > target_seconds:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(build_chunk_state(chunk_text, target_seconds, words_per_second, sentence_pause_seconds))
            current_sentences = [sentence]
            current_estimate = sentence_estimate
        else:
            current_sentences.append(sentence)
            current_estimate += sentence_estimate

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(build_chunk_state(chunk_text, target_seconds, words_per_second, sentence_pause_seconds))

    return chunks


def session_dir_for(name_stem: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / f"{name_stem}_{timestamp}_{secrets.token_hex(4)}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def copy_uploaded_or_local_file(uploaded_path: str | None, local_name: str | None, session_dir: Path) -> tuple[str, str]:
    if uploaded_path:
        source = Path(uploaded_path).resolve()
        destination = session_dir / source.name
        shutil.copy2(source, destination)
        return str(destination), source.name

    if local_name:
        source = (WORKSPACE / local_name).resolve()
        if not source.is_relative_to(WORKSPACE.resolve()):
            raise ValueError(f"Invalid local file path: {local_name}")
        if not source.exists():
            raise FileNotFoundError(f"Local file not found: {source}")
        return str(source), source.name

    raise ValueError("Please choose a local file or upload one.")


def build_empty_session() -> SessionState:
    """Create the default Gradio session state payload."""
    return {
        "session_dir": None,
        "source_name": None,
        "source_path": None,
        "reference_path": None,
        "reference_name": None,
        "target_chunk_seconds": DEFAULT_TARGET_CHUNK_SECONDS,
        "words_per_second": DEFAULT_WORDS_PER_SECOND,
        "sentence_pause_seconds": DEFAULT_SENTENCE_PAUSE_SECONDS,
        "merge_silence_seconds": DEFAULT_MERGE_SILENCE_SECONDS,
        "context_limit": DEFAULT_CONTEXT_LIMIT,
        "chunks": [],
        "current_index": 0,
        "merged_path": None,
    }


def current_chunk(session: SessionState) -> ChunkState | None:
    chunks = session.get("chunks", [])
    if not chunks:
        return None
    index = max(0, min(session.get("current_index", 0), len(chunks) - 1))
    session["current_index"] = index
    return chunks[index]


def persist_chunk_text(session: SessionState, text: str | None) -> None:
    chunk = current_chunk(session)
    if chunk is None or text is None:
        return

    normalized = normalize_text(text)
    if normalized == chunk["text"]:
        return

    chunk["text"] = normalized
    chunk["estimated_seconds"] = estimate_seconds(
        normalized,
        session["words_per_second"],
        session["sentence_pause_seconds"],
    )
    chunk["audio_path"] = None
    chunk["actual_seconds"] = None
    chunk["approved"] = False
    chunk["seed"] = None
    chunk["generation_count"] = 0
    chunk["oversized"] = chunk["estimated_seconds"] > session["target_chunk_seconds"]
    session["merged_path"] = None


def chunk_status_text(chunk: ChunkState) -> str:
    if chunk["approved"]:
        return "approved"
    if chunk["audio_path"]:
        return "generated"
    return "pending"


def chunk_choice_label(index: int, chunk: ChunkState) -> str:
    """Build a readable dropdown label for one chunk."""
    preview = chunk["text"][:54].strip()
    preview = preview + "..." if len(chunk["text"]) > 54 else preview
    return f"{index + 1:02d} | {chunk_status_text(chunk):9s} | est {chunk['estimated_seconds']:.1f}s | {preview}"


def chunk_choice_updates(session: SessionState):
    """Return a Gradio dropdown update for the current chunk list."""
    chunks = session.get("chunks", [])
    if not chunks:
        return gr.update(choices=[], value=None, interactive=False)

    choices = [(chunk_choice_label(index, chunk), index + 1) for index, chunk in enumerate(chunks)]
    return gr.update(choices=choices, value=session["current_index"] + 1, interactive=True)


def find_pending_index(session: SessionState, start_index: int, step: int) -> int | None:
    """Find the next chunk that still needs review in the requested direction."""
    chunks = session.get("chunks", [])
    search_range = range(start_index + step, len(chunks), step) if step > 0 else range(start_index + step, -1, step)
    for index in search_range:
        chunk = chunks[index]
        if not chunk["approved"] or not chunk["audio_path"]:
            return index
    return None


def overview_markdown(session: SessionState) -> str:
    chunks = session.get("chunks", [])
    if not chunks:
        return "No text file loaded yet."

    approved = sum(1 for chunk in chunks if chunk["approved"])
    generated = sum(1 for chunk in chunks if chunk["audio_path"])
    pending = len(chunks) - generated

    lines = [
        f"**Source**: `{session['source_name']}`",
        f"**Reference**: `{session['reference_name']}`",
        f"**Chunks**: {len(chunks)} | **Approved**: {approved} | **Generated**: {generated} | **Pending**: {pending}",
        f"**Target Chunk Length**: {session['target_chunk_seconds']:.1f}s | **Speech Rate Estimate**: {session['words_per_second']:.2f} words/s",
        "",
    ]

    for index, chunk in enumerate(chunks, start=1):
        pointer = "->" if index - 1 == session["current_index"] else "  "
        actual = f"{chunk['actual_seconds']:.2f}s" if chunk["actual_seconds"] is not None else "-"
        lines.append(
            f"{pointer} {index:02d}. {chunk_status_text(chunk)} | est {chunk['estimated_seconds']:.2f}s | actual {actual}"
        )

    return "\n".join(lines)


def chunk_detail_markdown(session: SessionState) -> str:
    chunk = current_chunk(session)
    if chunk is None:
        return "Load a text file to begin."

    details = [
        f"**Estimated Length**: {chunk['estimated_seconds']:.2f}s",
        f"**Actual Length**: {chunk['actual_seconds']:.2f}s" if chunk["actual_seconds"] is not None else "**Actual Length**: -",
        f"**Status**: {chunk_status_text(chunk)}",
        f"**Seed**: {chunk['seed']}" if chunk["seed"] is not None else "**Seed**: -",
        f"**Generations**: {chunk['generation_count']}",
    ]
    if chunk["oversized"]:
        details.append("**Note**: this chunk exceeds the target estimate because it keeps a full sentence intact.")
    details.append("Edits are saved when you navigate, generate, approve, or merge.")
    return "\n".join(details)


def render_ui(session: SessionState, message: str = ""):
    chunks = session.get("chunks", [])
    if not chunks:
        return (
            message or "Load a French text file to begin.",
            session,
            "No text file loaded yet.",
            chunk_choice_updates(session),
            "Chunk 0 / 0",
            "Load a text file to begin.",
            "",
            None,
            None,
            None,
            None,
            model_status_text(),
        )

    index = session["current_index"]
    chunk = chunks[index]
    merged_path = session.get("merged_path")
    return (
        message,
        session,
        overview_markdown(session),
        chunk_choice_updates(session),
        f"Chunk {index + 1} / {len(chunks)}",
        chunk_detail_markdown(session),
        chunk["text"],
        chunk["audio_path"],
        chunk["seed"],
        merged_path,
        merged_path,
        model_status_text(),
    )


def ensure_model(context_limit: int = DEFAULT_CONTEXT_LIMIT) -> VoxCPM:
    global MODEL_INSTANCE, MODEL_SAMPLE_RATE, MODEL_LOADING, MODEL_PRELOAD_ERROR, MODEL_CONTEXT_LIMIT

    if MODEL_INSTANCE is not None:
        if MODEL_CONTEXT_LIMIT == context_limit:
            return MODEL_INSTANCE
        unload_model("context-limit-changed")

    MODEL_LOADING = True
    MODEL_PRELOAD_ERROR = None
    cleanup_vram("gui-before-load")
    try:
        patch_voxcpm2_loader(context_limit)
        model_path = snapshot_download(repo_id=MODEL_ID, local_files_only=False)
        MODEL_INSTANCE = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
        MODEL_SAMPLE_RATE = MODEL_INSTANCE.tts_model.sample_rate
        MODEL_CONTEXT_LIMIT = context_limit
        return MODEL_INSTANCE
    except Exception as exc:
        MODEL_PRELOAD_ERROR = str(exc)
        raise
    finally:
        MODEL_LOADING = False


def model_status_text() -> str:
    if MODEL_LOADING:
        return "Model preload is running in the background. The first chunk generation should be faster once it finishes."
    if MODEL_PRELOAD_ERROR:
        return (
            "Background preload failed. You can still retry with the load button or by generating a chunk. "
            f"Last error: `{MODEL_PRELOAD_ERROR}`"
        )
    if MODEL_INSTANCE is None:
        if MODEL_PRELOAD_STARTED:
            return "Background preload has been scheduled, but the model is not ready yet."
        return "Model not loaded yet. It will auto-load on the first generation, or you can preload it now."
    device = getattr(MODEL_INSTANCE.tts_model, "device", "unknown")
    sample_rate = MODEL_SAMPLE_RATE or getattr(MODEL_INSTANCE.tts_model, "sample_rate", "unknown")
    return f"Model is loaded on `{device}` and kept resident for later chunk generations. Sample rate: `{sample_rate}`."


def unload_model(reason: str = "manual") -> str:
    global MODEL_INSTANCE, MODEL_SAMPLE_RATE, MODEL_LOADING, MODEL_PRELOAD_ERROR, MODEL_CONTEXT_LIMIT

    with MODEL_LOCK:
        old_model = MODEL_INSTANCE
        MODEL_INSTANCE = None
        MODEL_SAMPLE_RATE = None
        MODEL_CONTEXT_LIMIT = None
        MODEL_LOADING = False
        MODEL_PRELOAD_ERROR = None
        if old_model is not None:
            del old_model

    cleanup_vram(f"gui-after-unload-{reason}")
    return model_status_text()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preload_model(context_limit: int) -> tuple[str, str]:
    with MODEL_LOCK:
        if MODEL_INSTANCE is not None and MODEL_CONTEXT_LIMIT == int(context_limit):
            return model_status_text(), "Model is already loaded with requested context limit. Skipping load."
        ensure_model(int(context_limit))
    return model_status_text(), "Model preloaded successfully. Later chunk generations will reuse it."


def unload_model_action() -> tuple[str, str]:
    status = unload_model("button")
    return status, "Model unloaded and VRAM cleanup requested."


def background_preload_worker(context_limit: int) -> None:
    try:
        with MODEL_LOCK:
            ensure_model(int(context_limit))
    except Exception:
        pass


def start_background_preload(context_limit: int = DEFAULT_CONTEXT_LIMIT) -> None:
    global MODEL_PRELOAD_STARTED

    if MODEL_PRELOAD_STARTED or MODEL_INSTANCE is not None:
        return

    MODEL_PRELOAD_STARTED = True
    thread = threading.Thread(
        target=background_preload_worker,
        args=(int(context_limit),),
        name="voxcpm-background-preload",
        daemon=True,
    )
    thread.start()


def _shutdown_handler(signum=None, frame=None) -> None:
    try:
        unload_model(f"signal-{signum}" if signum is not None else "atexit")
    except Exception:
        pass
    if signum is not None:
        raise SystemExit(0)


def register_shutdown_handlers() -> None:
    atexit.register(_shutdown_handler)
    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signal_name, None)
        if sig is not None:
            signal.signal(sig, _shutdown_handler)


def load_text_into_session(
    selected_text_file: str | None,
    uploaded_text_file: str | None,
    selected_reference_file: str | None,
    uploaded_reference_file: str | None,
    target_chunk_seconds: float,
    words_per_second: float,
    sentence_pause_seconds: float,
    merge_silence_seconds: float,
    context_limit: int,
) -> tuple:
    if target_chunk_seconds <= 0:
        raise gr.Error("Target chunk seconds must be positive.")
    if words_per_second <= 0:
        raise gr.Error("Words per second must be positive.")
    if sentence_pause_seconds < 0:
        raise gr.Error("Sentence pause must be zero or positive.")
    if merge_silence_seconds < 0.1:
        raise gr.Error("Merge silence must be at least 0.1 seconds.")

    text_label = uploaded_text_file or selected_text_file
    source_stem = Path(text_label).stem if text_label else "session"
    session_dir = session_dir_for(source_stem)

    source_path, source_name = copy_uploaded_or_local_file(uploaded_text_file, selected_text_file, session_dir)
    reference_path, reference_name = copy_uploaded_or_local_file(
        uploaded_reference_file,
        selected_reference_file,
        session_dir,
    )

    text = read_text_file(source_path)
    chunks = chunk_text_for_target_duration(
        text,
        target_chunk_seconds,
        words_per_second,
        sentence_pause_seconds,
    )
    if not chunks:
        raise gr.Error("The text file appears to be empty after normalization.")

    session = build_empty_session()
    session.update(
        {
            "session_dir": str(session_dir),
            "source_name": source_name,
            "source_path": source_path,
            "reference_path": reference_path,
            "reference_name": reference_name,
            "target_chunk_seconds": target_chunk_seconds,
            "words_per_second": words_per_second,
            "sentence_pause_seconds": sentence_pause_seconds,
            "merge_silence_seconds": merge_silence_seconds,
            "context_limit": int(context_limit),
            "chunks": chunks,
            "current_index": 0,
            "merged_path": None,
        }
    )

    return render_ui(
        session,
        f"Loaded `{source_name}` into {len(chunks)} sentence-safe chunks using `{reference_name}`.",
    )


def go_to_chunk(session: SessionState, current_text: str, target_index: int) -> tuple:
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    session["current_index"] = max(0, min(int(target_index) - 1, len(session["chunks"]) - 1))
    return render_ui(session, f"Moved to chunk {session['current_index'] + 1}.")


def go_prev(session: SessionState, current_text: str) -> tuple:
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    session["current_index"] = max(0, session["current_index"] - 1)
    return render_ui(session, f"Moved to chunk {session['current_index'] + 1}.")


def go_next(session: SessionState, current_text: str) -> tuple:
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    session["current_index"] = min(len(session["chunks"]) - 1, session["current_index"] + 1)
    return render_ui(session, f"Moved to chunk {session['current_index'] + 1}.")


def go_to_pending(session: SessionState, current_text: str, step: int) -> tuple:
    """Jump to the next or previous chunk that still needs review."""
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    next_index = find_pending_index(session, session["current_index"], step)
    if next_index is None:
        direction = "next" if step > 0 else "previous"
        return render_ui(session, f"No {direction} pending chunk found.")
    session["current_index"] = next_index
    return render_ui(session, f"Jumped to pending chunk {session['current_index'] + 1}.")


def validate_generation_inputs(session: SessionState, timesteps: int, max_len: int) -> None:
    """Validate the common generation parameters shared by single and bulk actions."""
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    if timesteps <= 0:
        raise gr.Error("Timesteps must be positive.")
    if max_len <= 0:
        raise gr.Error("Max length must be positive.")


def resolve_seed(seed_value: int | None, force_new_seed: bool) -> int:
    """Choose a deterministic seed for the next generation."""
    if force_new_seed or seed_value in (None, "", 0):
        return safe_random_seed()
    return int(seed_value)


def generate_and_store_chunk_audio(
    session: SessionState,
    model: VoxCPM,
    chunk_index: int,
    cfg_value: float,
    timesteps: int,
    max_len: int,
    seed: int,
) -> None:
    """Run VoxCPM for one chunk and persist the resulting WAV plus chunk metadata."""
    chunk = session["chunks"][chunk_index]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.inference_mode():
        wav = model.generate(
            text=chunk["text"],
            reference_wav_path=session["reference_path"],
            cfg_value=float(cfg_value),
            inference_timesteps=int(timesteps),
            max_len=int(max_len),
        )

    session_dir = Path(cast(str, session["session_dir"]))
    version = chunk["generation_count"] + 1
    output_path = session_dir / f"chunk_{chunk_index + 1:03d}_v{version:02d}_seed{seed}.wav"
    sf.write(output_path, wav, model.tts_model.sample_rate)

    chunk["audio_path"] = str(output_path)
    chunk["actual_seconds"] = round(len(wav) / model.tts_model.sample_rate, 2)
    chunk["approved"] = False
    chunk["seed"] = seed
    chunk["generation_count"] = version
    session["merged_path"] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_current_chunk(
    session: SessionState,
    current_text: str,
    cfg_value: float,
    timesteps: int,
    max_len: int,
    seed_value: int | None,
    force_new_seed: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate one take for the currently selected chunk."""
    validate_generation_inputs(session, timesteps, max_len)

    persist_chunk_text(session, current_text)
    chunk = current_chunk(session)
    assert chunk is not None

    if not chunk["text"].strip():
        raise gr.Error("The current chunk is empty.")

    # Yield intermediate state to clear audio player and reset playhead
    chunk["audio_path"] = None
    yield render_ui(session, "Preparing generation...")

    seed = resolve_seed(seed_value, force_new_seed)
    progress(0.0, desc="Loading VoxCPM2 model")

    with MODEL_LOCK:
        model = ensure_model(session["context_limit"])
        progress(0.2, desc="Generating chunk audio")
        generate_and_store_chunk_audio(
            session,
            model,
            session["current_index"],
            cfg_value,
            timesteps,
            max_len,
            seed,
        )

    yield render_ui(
        session,
        f"Generated chunk {session['current_index'] + 1} with seed {seed}. Approve it when you like the take.",
    )


def approve_current(session: SessionState, current_text: str, move_next: bool) -> tuple:
    """Approve the current chunk and optionally advance to the next one."""
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    chunk = current_chunk(session)
    assert chunk is not None

    if not chunk["audio_path"]:
        raise gr.Error("Generate audio for the current chunk before approving it.")

    chunk["approved"] = True
    message = f"Approved chunk {session['current_index'] + 1}."
    if move_next:
        next_pending = find_pending_index(session, session["current_index"], 1)
        if next_pending is not None:
            session["current_index"] = next_pending
            message += f" Moved to next pending chunk {session['current_index'] + 1}."
        elif session["current_index"] < len(session["chunks"]) - 1:
            session["current_index"] += 1
            message += f" No later pending chunk found, so moved to chunk {session['current_index'] + 1}."

    return render_ui(session, message)


def mark_current_pending(session: SessionState, current_text: str) -> tuple:
    """Return the current chunk to a review-needed state."""
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    persist_chunk_text(session, current_text)
    chunk = current_chunk(session)
    assert chunk is not None

    chunk["approved"] = False
    session["merged_path"] = None
    return render_ui(session, f"Chunk {session['current_index'] + 1} marked as pending.")


def merge_approved_chunks(
    session: SessionState,
    current_text: str,
    merge_silence_seconds: float,
) -> tuple:
    """Merge approved chunk WAV files into one export with fixed silence between chunks."""
    if not session.get("chunks"):
        raise gr.Error("Load a text file first.")
    if merge_silence_seconds < 0.1:
        raise gr.Error("Merge silence must be at least 0.1 seconds.")

    persist_chunk_text(session, current_text)

    approved_indices = [
        index + 1
        for index, chunk in enumerate(session["chunks"])
        if chunk["approved"] and chunk["audio_path"]
    ]
    if not approved_indices:
        raise gr.Error("No approved chunks with audio are available to merge.")

    merged_parts: list[np.ndarray] = []
    sample_rate: int | None = None

    kept_count = 0
    for index, chunk in enumerate(session["chunks"]):
        if not chunk["approved"] or not chunk["audio_path"]:
            continue
        audio, chunk_sr = sf.read(chunk["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = chunk_sr
        elif chunk_sr != sample_rate:
            raise gr.Error("Chunk sample rates do not match; merge aborted.")
        merged_parts.append(audio.astype(np.float32))
        kept_count += 1
        if kept_count < len(approved_indices):
            silence_samples = int(sample_rate * merge_silence_seconds)
            merged_parts.append(np.zeros(silence_samples, dtype=np.float32))

    assert sample_rate is not None
    merged_audio = np.concatenate(merged_parts)
    output_path = Path(session["session_dir"]) / f"merged_{Path(session['source_name']).stem}.wav"
    sf.write(output_path, merged_audio, sample_rate)
    session["merged_path"] = str(output_path)
    session["merge_silence_seconds"] = merge_silence_seconds

    duration = len(merged_audio) / sample_rate
    skipped_count = len(session["chunks"]) - len(approved_indices)
    kept_preview = ", ".join(str(item) for item in approved_indices[:10])
    kept_suffix = "..." if len(approved_indices) > 10 else ""
    return render_ui(
        session,
        f"Merged {len(approved_indices)} approved chunks into `{output_path.name}` ({duration:.2f}s). "
        f"Skipped {skipped_count} missing/unapproved chunks. Included: {kept_preview}{kept_suffix}",
    )


def generate_all_pending_chunks(
    session: SessionState,
    current_text: str,
    cfg_value: float,
    timesteps: int,
    max_len: int,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate fresh takes for every chunk that still needs review."""
    validate_generation_inputs(session, timesteps, max_len)

    persist_chunk_text(session, current_text)
    pending_indices = [
        index
        for index, chunk in enumerate(session["chunks"])
        if (not chunk["approved"] or not chunk["audio_path"]) and chunk["text"].strip()
    ]

    if not pending_indices:
        yield render_ui(session, "No pending chunks to generate.")
        return

    current = current_chunk(session)
    if current:
        current["audio_path"] = None
    yield render_ui(session, "Preparing batch generation...")

    with MODEL_LOCK:
        progress(0.0, desc="Loading VoxCPM2 model")
        model = ensure_model(session["context_limit"])

        for step, chunk_index in enumerate(pending_indices, start=1):
            chunk = session["chunks"][chunk_index]
            if not chunk["text"].strip():
                continue

            seed = safe_random_seed()
            progress(
                step / max(1, len(pending_indices)),
                desc=f"Generating chunk {chunk_index + 1}/{len(session['chunks'])}",
            )
            generate_and_store_chunk_audio(
                session,
                model,
                chunk_index,
                cfg_value,
                timesteps,
                max_len,
                seed,
            )

    session["current_index"] = pending_indices[0]
    yield render_ui(
        session,
        f"Generated {len(pending_indices)} pending chunks. Review and approve the takes you want to keep.",
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VoxCPM Chunk Studio", fill_width=True) as demo:
        session_state = gr.State(build_empty_session())

        gr.Markdown(
            """
            # VoxCPM Chunk Studio
            French-first chunk review for VoxCPM2.

            Load a text file, split it into sentence-safe chunks around the target duration, generate audio for each chunk,
            approve the takes you like, and merge the approved chunks into one export with a short silence between them.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_file_choice = gr.Dropdown(
                    label="Workspace Text File",
                    choices=workspace_text_files(),
                    value="another_test.txt" if "another_test.txt" in workspace_text_files() else None,
                )
                text_file_upload = gr.File(label="Upload Text File", type="filepath")
                reference_choice = gr.Dropdown(
                    label="Reference Audio",
                    choices=workspace_reference_files(),
                    value=default_reference_name(),
                )
                reference_upload = gr.File(label="Upload Reference Audio", type="filepath")
                load_button = gr.Button("Load And Chunk", variant="primary")

            with gr.Column(scale=1):
                target_chunk_seconds = gr.Slider(
                    minimum=6,
                    maximum=14,
                    step=0.5,
                    value=DEFAULT_TARGET_CHUNK_SECONDS,
                    label="Target Chunk Length (seconds)",
                )
                words_per_second = gr.Slider(
                    minimum=2.4,
                    maximum=3.8,
                    step=0.05,
                    value=DEFAULT_WORDS_PER_SECOND,
                    label="Estimated Speech Rate (words/second)",
                )
                sentence_pause_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=0.4,
                    step=0.01,
                    value=DEFAULT_SENTENCE_PAUSE_SECONDS,
                    label="Sentence Pause Budget (seconds)",
                )
                merge_silence_seconds = gr.Slider(
                    minimum=0.1,
                    maximum=0.5,
                    step=0.01,
                    value=DEFAULT_MERGE_SILENCE_SECONDS,
                    label="Silence Between Exported Chunks (seconds)",
                )

        with gr.Accordion("Advanced Generation Settings", open=False):
            with gr.Row():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    step=0.1,
                    value=DEFAULT_CFG_VALUE,
                    label="CFG Value",
                )
                timesteps = gr.Slider(
                    minimum=4,
                    maximum=12,
                    step=1,
                    value=DEFAULT_TIMESTEPS,
                    label="Inference Timesteps",
                )
                max_len = gr.Slider(
                    minimum=128,
                    maximum=384,
                    step=16,
                    value=DEFAULT_MAX_LEN,
                    label="Max Length",
                )
                context_limit = gr.Slider(
                    minimum=384,
                    maximum=768,
                    step=32,
                    value=DEFAULT_CONTEXT_LIMIT,
                    label="Internal Context Limit",
                )

        with gr.Row():
            load_model_button = gr.Button("Load Model Into VRAM")
            unload_model_button = gr.Button("Unload Model And Free VRAM")

        model_status = gr.Markdown(model_status_text())
        operation_status = gr.Markdown("Load a French text file to begin.")
        demo.load(fn=model_status_text, inputs=None, outputs=[model_status])

        with gr.Row():
            overview = gr.Markdown("No text file loaded yet.")
            with gr.Column(scale=1):
                chunk_selector = gr.Dropdown(
                    label="Chunk Selector",
                    choices=[],
                    value=None,
                    interactive=False,
                )
                chunk_title = gr.Markdown("Chunk 0 / 0")
                chunk_details = gr.Markdown("Load a text file to begin.")

        chunk_textbox = gr.Textbox(
            label="Current Chunk Text",
            lines=10,
            max_lines=16,
            placeholder="The selected chunk text appears here.",
        )

        with gr.Row():
            with gr.Group():
                gr.Markdown("### Navigation")
                with gr.Row():
                    prev_button = gr.Button("Previous Chunk")
                    next_button = gr.Button("Next Chunk")
                    prev_pending_button = gr.Button("Previous Pending")
                    next_pending_button = gr.Button("Next Pending")
            with gr.Group():
                gr.Markdown("### Generation")
                with gr.Row():
                    generate_button = gr.Button("Generate Current Chunk", variant="primary")
                    regenerate_button = gr.Button("Generate New Take")
                    generate_pending_button = gr.Button("Generate All Pending Chunks")
            with gr.Group():
                gr.Markdown("### Approval")
                with gr.Row():
                    approve_button = gr.Button("Approve Current Chunk", variant="secondary")
                    approve_next_button = gr.Button("Approve And Next Pending")
                    pending_button = gr.Button("Mark Pending")

        seed_number = gr.Number(label="Seed For Current Chunk", precision=0)
        current_audio = gr.Audio(label="Current Chunk Audio", type="filepath")

        with gr.Row():
            merge_button = gr.Button("Merge Approved Chunks", variant="primary")
            merged_audio = gr.Audio(label="Merged Audio Preview", type="filepath")
            merged_file = gr.File(label="Merged Export")

        shared_outputs = [
            operation_status,
            session_state,
            overview,
            chunk_selector,
            chunk_title,
            chunk_details,
            chunk_textbox,
            current_audio,
            seed_number,
            merged_audio,
            merged_file,
            model_status,
        ]

        load_model_button.click(
            fn=preload_model,
            inputs=[context_limit],
            outputs=[model_status, operation_status],
        )

        unload_model_button.click(
            fn=unload_model_action,
            inputs=[],
            outputs=[model_status, operation_status],
        )

        load_button.click(
            fn=load_text_into_session,
            inputs=[
                text_file_choice,
                text_file_upload,
                reference_choice,
                reference_upload,
                target_chunk_seconds,
                words_per_second,
                sentence_pause_seconds,
                merge_silence_seconds,
                context_limit,
            ],
            outputs=shared_outputs,
        )

        chunk_selector.change(
            fn=go_to_chunk,
            inputs=[session_state, chunk_textbox, chunk_selector],
            outputs=shared_outputs,
        )

        prev_button.click(
            fn=go_prev,
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        next_button.click(
            fn=go_next,
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        prev_pending_button.click(
            fn=lambda session, text: go_to_pending(session, text, -1),
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        next_pending_button.click(
            fn=lambda session, text: go_to_pending(session, text, 1),
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        def _generate_current(session, text, cfg, steps, max_len_value, seed):
            yield from generate_current_chunk(session, text, cfg, steps, max_len_value, seed, False)

        def _regenerate_current(session, text, cfg, steps, max_len_value, seed):
            yield from generate_current_chunk(session, text, cfg, steps, max_len_value, seed, True)

        generate_button.click(
            fn=_generate_current,
            inputs=[session_state, chunk_textbox, cfg_value, timesteps, max_len, seed_number],
            outputs=shared_outputs,
            show_progress="minimal",
        )

        regenerate_button.click(
            fn=_regenerate_current,
            inputs=[session_state, chunk_textbox, cfg_value, timesteps, max_len, seed_number],
            outputs=shared_outputs,
            show_progress="minimal",
        )

        def _generate_all_pending(session, text, cfg, steps, max_len_value):
            yield from generate_all_pending_chunks(session, text, cfg, steps, max_len_value)

        generate_pending_button.click(
            fn=_generate_all_pending,
            inputs=[session_state, chunk_textbox, cfg_value, timesteps, max_len],
            outputs=shared_outputs,
            show_progress="minimal",
        )

        approve_button.click(
            fn=lambda session, text: approve_current(session, text, False),
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        approve_next_button.click(
            fn=lambda session, text: approve_current(session, text, True),
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        pending_button.click(
            fn=mark_current_pending,
            inputs=[session_state, chunk_textbox],
            outputs=shared_outputs,
        )

        merge_button.click(
            fn=merge_approved_chunks,
            inputs=[session_state, chunk_textbox, merge_silence_seconds],
            outputs=shared_outputs,
        )

    return demo


demo = build_demo()


def main() -> None:
    register_shutdown_handlers()
    start_background_preload(DEFAULT_CONTEXT_LIMIT)
    try:
        demo.queue(default_concurrency_limit=1).launch()
    finally:
        unload_model("main-finally")


if __name__ == "__main__":
    main()
