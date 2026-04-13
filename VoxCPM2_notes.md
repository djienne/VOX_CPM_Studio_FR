# VoxCPM2 Notes

Last updated: 2026-04-13

This file is a practical, source-linked summary of VoxCPM2 based on the official Hugging Face model card, the OpenBMB GitHub repository, the VoxCPM documentation site, and the local Windows/GPU environment on this machine. It is not a verbatim mirror of those sources.

## Local project focus

For this workspace, the main practical focus is French:

- default example text should be French
- French TTS quality matters more than generic multilingual demonstrations
- `ref_short.mp3` and `ref_long.mp3` should be treated as French cloning references
- multilingual features still matter, but French-first examples are preferred

## Official sources

- Hugging Face model card: <https://huggingface.co/openbmb/VoxCPM2>
- GitHub repository: <https://github.com/OpenBMB/VoxCPM>
- Quick Start docs: <https://voxcpm.readthedocs.io/en/latest/quickstart.html>
- Installation docs: <https://voxcpm.readthedocs.io/en/latest/installation.html>
- FAQ / Troubleshooting: <https://voxcpm.readthedocs.io/en/latest/faq.html>
- Usage guide: <https://voxcpm.readthedocs.io/en/latest/usage.html>

## What VoxCPM2 is

VoxCPM2 is OpenBMB's current flagship open-source text-to-speech model. The official materials describe it as a tokenizer-free diffusion autoregressive TTS system built on a MiniCPM-4 backbone. The Hugging Face model card and GitHub README both present it as a 2B-parameter release trained on more than 2 million hours of multilingual speech data.

According to the official model card and repo, the main advertised capabilities are:

- Multilingual TTS across 30 languages without language tags.
- Voice design from natural-language voice descriptions.
- Voice cloning from short reference audio.
- "Ultimate cloning" when both reference audio and transcript are provided.
- 48 kHz output, with 16 kHz reference audio accepted for cloning workflows.
- Streaming generation support.
- Fine-tuning support, including LoRA.

The supported languages listed by the official sources are:

- Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese.

The model card and repo also mention several Chinese dialects, including Cantonese, Sichuanese, Wu, Northeastern Mandarin, Henan, Shaanxi, Shandong, Tianjin, and Minnan.

## Model and runtime details from the official docs

The official Hugging Face card summarizes the current release roughly as follows:

- Architecture: tokenizer-free diffusion autoregressive TTS.
- Parameter scale: 2B.
- Sample rate: 48 kHz output.
- Default dtype: bfloat16.
- Max sequence length: 8192 tokens.
- Reported VRAM target: about 8 GB.

The installation docs state these runtime requirements or recommendations:

- Python: 3.10 to 3.12 recommended, with 3.10 to 3.11 most tested.
- PyTorch: 2.5.0 or newer.
- CUDA: optional, but 12.0+ for NVIDIA GPU acceleration.
- Disk: several GB for model weights, depending on the checkpoint.

The docs also state that device selection can be automatic or explicit:

- Auto fallback order: `cuda -> mps -> cpu`
- Explicit CUDA examples: `device="cuda"` or `device="cuda:0"`
- For CPU or platform-specific issues, `optimize=False` is the documented safe fallback.

## Official install paths

The fastest documented install path is:

```bash
pip install voxcpm
```

The web demo requires a source checkout in addition to the package install:

```bash
git clone https://github.com/OpenBMB/VoxCPM.git
cd VoxCPM
pip install -e .
python app.py
```

The docs say the first run downloads model weights automatically from Hugging Face. The GitHub README also shows a mirror path using ModelScope when Hugging Face access is inconvenient.

## Official minimal usage patterns

### Python API

The simplest documented path is loading `openbmb/VoxCPM2`, calling `generate(...)`, and writing the resulting waveform to disk with `soundfile`.

For this Windows machine, the safer adaptation is to force CUDA explicitly and disable compile-time optimization:

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    device="cuda:0",
    optimize=False,
    load_denoiser=False,
)

wav = model.generate(
    text="This is a VoxCPM2 GPU smoke test on Windows.",
    cfg_value=2.0,
    inference_timesteps=10,
)

sf.write("demo.wav", wav, model.tts_model.sample_rate)
```

### CLI

The Quick Start docs show CLI commands such as:

- `voxcpm design --text "Hello from VoxCPM!" --output out.wav`
- `voxcpm clone --text "..." --reference-audio path/to/voice.wav --output out.wav --denoise`

For this machine, an adjusted Windows/GPU test command would be:

```bash
voxcpm design --text "Hello from VoxCPM" --device cuda:0 --no-optimize --output out.wav
```

## Additional online findings

These points come from the official docs site plus maintainer comments and issue threads in the official `OpenBMB/VoxCPM` repository and the official Hugging Face discussion board.

### Usage guide details that matter in practice

From the official Usage Guide:

- `reference_wav_path` is the normal VoxCPM 2 cloning path and does not require a transcript.
- `prompt_wav_path` must be paired with `prompt_text`.
- A practical reference range is 5 to 30 seconds.
- MP3, WAV, and FLAC are all acceptable as long as torchaudio can load them.
- Very short text can sound weak; text that naturally produces a few seconds of speech is more stable.
- Long text can trigger buzzing, speed drift, OOM, or generations that never stop. The docs recommend splitting long text into shorter segments and concatenating the outputs.
- Lowering `cfg_value` toward `1.5` to `1.6` can help if outputs become noisy.
- `denoise=True` is for noisy references; if the reference is already clean, `denoise=False` often preserves voice characteristics better.

### Important maintainer comments from official issues

From official repo issues and maintainer comments:

- `prompt_wav_path` is effectively the continuation / transcript-aligned path, and a maintainer stated that this prompt-audio route currently supports Chinese and English. For French work, `reference_wav_path` is the safer first choice.
- A maintainer also said the denoiser can become a bottleneck when it falls back to CPU. Turning denoising off can improve speed.
- Another maintainer comment says the warm-up step happens during model load, not before every later generation call. That is useful if the model stays resident in a long-running process.
- In one official issue, a maintainer reported that on a 4090, generating about 30 seconds of speech stayed under roughly 5 GB VRAM. That is not a guarantee for this machine, but it suggests runtime VRAM can be lower than the full model-card estimate depending on the exact path and output length.

### Hugging Face community notes worth knowing

From the official Hugging Face model discussions:

- There are already user requests for 8-bit and 16-bit quantization, but the official model card and docs do not yet document a supported quantized inference workflow for VoxCPM2.
- One user reported pacing issues in ultimate cloning and another user said changing the seed helped. I treat that as a community observation rather than an official guarantee.

## French-specific implications for this workspace

Because the refs in this folder are French and the maintainer comments say prompt-audio continuation is mainly a Chinese/English path, the best French-first testing order is:

1. French voice design or plain French TTS with no reference audio.
2. French controllable cloning using `reference_wav_path` only.
3. Only try transcript-aligned prompt mode later if we have a strong reason and can verify the transcript path behaves acceptably for French.

For the French cloning references here:

- `ref_short.mp3` is likely a good first attempt if it is at least about 5 seconds and clean.
- If `ref_short.mp3` is too short, clip a clean 5 to 15 second section from `ref_long.mp3` and use that as an intermediate French reference file.
- Avoid turning on denoising unless the reference is actually noisy.

## Windows and troubleshooting notes from the official docs

The FAQ page is especially important for Windows.

### 1. Triton / `torch.compile` on Windows

The docs call out Triton-related failures on Windows as a known pain point. Their recommended safe fallback is to skip compile acceleration entirely:

```python
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", optimize=False)
```

The same idea applies from the CLI with `--no-optimize`.

The FAQ also lists a PyTorch/Triton compatibility table for Windows community builds:

- PyTorch 2.4 / 2.5 -> Triton 3.1
- PyTorch 2.6 -> Triton 3.2
- PyTorch 2.7 -> Triton 3.3
- PyTorch 2.8 -> Triton 3.4

### 2. `torchcodec` / FFmpeg issues

The docs say cloning workflows may fail if `torchcodec` or FFmpeg is missing or misconfigured. Their recommended actions are:

- Install FFmpeg and add it to `PATH` on Windows.
- Ensure `torchcodec` is installed.
- If needed, force the torchaudio backend to `soundfile`.

### 3. Mac / MPS support

The official FAQ says MPS is supported on Apple Silicon, but CPU fallback may still be needed for some runtime paths. That is less relevant to this Windows machine, but useful if the setup is moved later.

## GitHub repo structure

The repository currently exposes these top-level areas in the GitHub file tree:

- `conf`
- `examples`
- `scripts`
- `src/voxcpm`
- `tests`
- `app.py`
- `pyproject.toml`

That suggests the repo is set up for:

- library usage via `src/voxcpm`
- example scripts
- web demo execution through `app.py`
- training and fine-tuning helpers in `scripts`
- config-driven workflows in `conf`

## Local environment status on this PC

Environment created for this task:

- Conda env name: `voxcpm`
- Python: `3.10.20`
- Python path: `C:\Users\david\miniconda3\envs\voxcpm\python.exe`

Installed package versions confirmed locally:

- `voxcpm 2.0.2`
- `torch 2.7.1+cu128`
- `torchaudio 2.7.1+cu128`
- `soundfile 0.13.1`

GPU state confirmed locally on 2026-04-13:

- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Total VRAM: `8188 MiB`
- VRAM already in use before model load: about `2892 MiB`
- Driver version: `595.79`
- PyTorch CUDA runtime in env: `12.8`
- `torch.cuda.is_available()`: `True`

Reference audio files found locally:

- `ref_short.mp3`: about `18.47s` at `44.1kHz`
- `ref_long.mp3`: about `813.96s` at `44.1kHz`

For French cloning, `ref_short.mp3` already sits inside the docs’ practical 5 to 30 second range, so it is the best first cloning reference.

## Successful local runs

Successful French-first runs completed on 2026-04-13 with the workspace script [run_voxcpm2_french.py](C:/Users/david/Desktop/VOXCPM/run_voxcpm2_french.py):

- Plain French TTS succeeded on GPU and produced [sortie_fr_test.wav](C:/Users/david/Desktop/VOXCPM/sortie_fr_test.wav)
- French voice design succeeded on GPU and produced [sortie_fr_voice_design.wav](C:/Users/david/Desktop/VOXCPM/sortie_fr_voice_design.wav)
- French reference-only cloning with `ref_short.mp3` succeeded on GPU and produced [sortie_fr_clone_ref_short.wav](C:/Users/david/Desktop/VOXCPM/sortie_fr_clone_ref_short.wav)

Observed output details:

- `sortie_fr_test.wav`: `48kHz`, about `4.16s`
- `sortie_fr_voice_design.wav`: `48kHz`, about `4.96s`
- `sortie_fr_clone_ref_short.wav`: `48kHz`, about `5.28s`

Observed GPU behavior from the successful runs:

- Before load: about `7075 MiB` free
- After load: about `1250 MiB` free
- Allocated by PyTorch after load: about `4434 MiB`
- Cleanup after run returned GPU memory to nearly the initial state

The runs used a local loader patch that reduced the internal `max_length` from `8192` to `512` and kept the AudioVAE on CPU while the main model stayed on CUDA. This is not an official documented mode, but it worked locally and appears to be the reason the 8 GB GPU run was viable.

## Local install commands already used

These are the commands that were used successfully for the clean Conda setup:

```bash
conda create -y -n voxcpm python=3.10
C:\Users\david\miniconda3\envs\voxcpm\python.exe -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
C:\Users\david\miniconda3\envs\voxcpm\python.exe -m pip install voxcpm soundfile
```

## Important practical risk on this machine

The official model card says VoxCPM2 is around the 8 GB VRAM class. This PC has an 8 GB RTX 4060 Ti, but nearly 2.9 GB was already in use before loading the model because the GPU is also driving the desktop. That means:

- a pure GPU run may still fail with out-of-memory depending on what else is open
- closing GPU-heavy apps before the first test is a good idea
- using `load_denoiser=False` and `optimize=False` is the safest first attempt
- using shorter French text and a reduced internal context length is a sensible 8 GB strategy because the docs warn that long text increases KV-cache pressure

## Suggested next command to try

If the environment is activated:

```bash
conda activate voxcpm
python smoke_test_voxcpm2.py
```

If not:

```bash
C:\Users\david\miniconda3\envs\voxcpm\python.exe smoke_test_voxcpm2.py
```

## Sources

- Hugging Face model card: <https://huggingface.co/openbmb/VoxCPM2>
- Hugging Face discussions: <https://huggingface.co/openbmb/VoxCPM2/discussions>
- GitHub repo / README: <https://github.com/OpenBMB/VoxCPM>
- Official issue #36 on Triton / compile setup: <https://github.com/OpenBMB/VoxCPM/issues/36>
- Official issue #65 on prompt/reference audio behavior: <https://github.com/OpenBMB/VoxCPM/issues/65>
- Official issue #67 on memory / warm-up / denoiser behavior: <https://github.com/OpenBMB/VoxCPM/issues/67>
- Official issue #86 on Windows torchcodec pain points: <https://github.com/OpenBMB/VoxCPM/issues/86>
- Official issue #119 on torchcodec / FFmpeg errors: <https://github.com/OpenBMB/VoxCPM/issues/119>
- Quick Start docs: <https://voxcpm.readthedocs.io/en/latest/quickstart.html>
- Installation docs: <https://voxcpm.readthedocs.io/en/latest/installation.html>
- Usage Guide: <https://voxcpm.readthedocs.io/en/latest/usage_guide.html>
- Version History: <https://voxcpm.readthedocs.io/en/latest/models/version_history.html>
- Changelog: <https://voxcpm.readthedocs.io/en/latest/reference/changelog.html>
- Cookbook: <https://voxcpm.readthedocs.io/en/latest/cookbook.html>
- FAQ / Troubleshooting: <https://voxcpm.readthedocs.io/en/latest/faq.html>
