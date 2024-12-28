from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", low_cpu_mem_usage=True)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(16_000))

sample = next(iter(dataset))
inputs = processor(sample["audio"]["array"], padding=True, truncation=False, return_attention_mask=True, return_tensors="pt")

outputs = model.generate(**inputs, return_segments=True)

print(outputs)