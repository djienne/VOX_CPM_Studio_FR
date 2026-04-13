# Workspace Guidance

## Main Focus

- Default to French for text-to-speech prompts, smoke tests, and example outputs.
- Prioritize French voices, French voice cloning, and French speech quality unless the user explicitly asks for another language.
- Treat `ref_short.mp3` and `ref_long.mp3` as French reference recordings.
- When choosing between a generic multilingual example and a French example, use French.

## Environment

- Preferred runtime environment: Conda env `voxcpm`
- Preferred model target: `openbmb/VoxCPM2`
- On this machine, keep Windows GPU constraints in mind and favor VRAM-conscious settings first.

## Practical Defaults

- Start with short French test text before longer generations.
- For Windows, prefer `optimize=False` unless there is a specific reason to test `torch.compile`.
- Clean VRAM before loading and after finishing a run.
- If cloning references are unsuitable, create a shorter intermediate French reference clip rather than forcing an overly long input.
