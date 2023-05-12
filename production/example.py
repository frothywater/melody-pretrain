from model import MelodyGLM


def melody_generation():
    checkpoint_path = "melodyglm_base_completion.ckpt"
    model = MelodyGLM.load_from_checkpoint(checkpoint_path, device="cuda:0")

    # Generate melody from scratch, with given sampling strategy
    result_midi = model.complete_melody(max_bar_length=32, temperature=0.9, top_k=10)

    # Or generate melody with given prompt
    result_midi = model.complete_melody("prompt.mid", max_bar_length=32, temperature=0.9, top_k=10)
    result_midi.dump("result.mid")


def melody_inpainting():
    checkpoint_path = "melodyglm_base_inpainting.ckpt"
    model = MelodyGLM.load_from_checkpoint(checkpoint_path, device="cuda:0")

    # Inpaint melody with given prompt and bar range [start, end)
    result_midi = model.inpaint_melody("prompt.mid", bar_range=(8, 16), temperature=0.9, top_k=10)

    # Or with given note range [start, end)
    result_midi = model.inpaint_melody("prompt.mid", note_range=(10, 20), temperature=0.9, top_k=10)

    result_midi.dump("inpainting.mid")


melody_generation()
melody_inpainting()
