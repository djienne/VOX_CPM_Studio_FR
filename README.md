# VoxCPM Studio FR

A French-first chunk review studio for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), built with Gradio.

This tool helps you process large text files into high-quality, natural-sounding Text-to-Speech (TTS) audio. It intelligently chunks text around target durations (while respecting sentence boundaries and French abbreviations), generates audio takes for each chunk, allows you to review and approve the best takes, and finally merges them into a single coherent audio file.

## Features

- **Smart Text Chunking:** Automatically splits text into sentence-safe chunks near a target duration. It includes specialized logic to avoid breaking sentences on common French abbreviations (like `M.`, `p.ex.`, `N.B.`, etc.).
- **Interactive Review Studio:** Generate, listen, and approve takes chunk by chunk. Keep track of what is pending and what is finished.
- **Background Preloading:** The heavy VoxCPM2 model is loaded into VRAM asynchronously to keep the UI snappy.
- **Audio Merging:** Stitch all your approved chunks together into one seamless export with configurable silence gaps.
- **Advanced Generation Controls:** Tweak CFG, inference timesteps, max sequence length, and internal context limits on the fly.
- **Playhead Reset:** Generating new takes cleanly resets the audio player so you always hear the fresh take from the beginning.

## Setup & Requirements

- Python 3.10+ (Anaconda or Miniconda recommended)
- An NVIDIA GPU with at least 8 GB VRAM

### Installation

1. **Create and activate a new Anaconda environment:**
   ```bash
   conda create -y -n voxcpm python=3.10
   conda activate voxcpm
   ```

2. **Install PyTorch with CUDA support:**
   *Make sure to choose the CUDA version that matches your system. Example for CUDA 12.8:*
   ```bash
   pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install the studio dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*Note: Ensure you have [FFmpeg](https://ffmpeg.org/) installed and added to your system `PATH`, as it is required by the underlying audio processing libraries.*

## Usage

You can start the studio using the provided batch file:

```bat
start_voxcpm_chunk_studio.bat
```

Or run the Python script directly if your environment is already activated:

```bash
python voxcpm_chunk_studio.py
```

### Workflow

1. **Load:** Select or upload a French text (`.txt`) file and a reference audio file (`.wav`, `.mp3`, etc.). Adjust your target chunk length and expected speech rate, then click **Load And Chunk**.
2. **Generate:** Use the **Generation** buttons to create audio for the current chunk. If you don't like a take, hit **Generate New Take**. To speed things up, you can click **Generate All Pending Chunks** to process everything in a batch.
3. **Approve:** Listen to the generated audio. If it sounds good, click **Approve Current Chunk**.
4. **Merge:** Once all chunks are approved, click **Merge Approved Chunks** to export the final stitched audio file.

Outputs and intermediate chunk files are automatically saved in the `gui_outputs/` directory within timestamped session folders.

## Recent Improvements

- **Ergonomics:** Logical UI grouping for Navigation, Generation, and Approval.
- **Streaming Updates:** Minimal progress bars to prevent UI flashing and dimming during generation.
- **Thread Safety:** Robust model loading logic (using `RLock`) that prevents accidental double-loads and safely manages context limit changes.
- **State Management:** Fixed edge cases where stale merged files could persist after a chunk's approval status was modified.
