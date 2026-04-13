import gradio as gr
import time

def test_audio():
    # simulate some work
    yield gr.update(value=None)
    time.sleep(1)
    # return a dummy audio (just use some path, even if it doesn't exist, to see if it clears)
    yield gr.update(value="ref_short.mp3")

with gr.Blocks() as demo:
    btn = gr.Button("Generate")
    audio = gr.Audio("ref_long.mp3", type="filepath")
    btn.click(fn=test_audio, inputs=[], outputs=[audio])

demo.launch(server_port=7865, prevent_thread_lock=True)
