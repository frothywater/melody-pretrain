from model import MelodyGLM

checkpoint_path = "experiment/production/melodyglm_finetuned_completion_small.ckpt"
model = MelodyGLM.load_from_checkpoint(checkpoint_path, device="cuda:0")
print("Model loaded successfully.")

# Generate melody from scratch, with given sampling strategy
result_midi = model.complete_melody(max_bar_length=32, temperature=1.2, top_k=5)
result_midi.dump("experiment/production/result.mid")

# Or generate melody with given prompt
# result_midi = model.complete_melody("prompt.mid", max_bar_length=32, temperature=1.2, top_k=5)
