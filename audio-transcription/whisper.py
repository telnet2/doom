import sys
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_audio_file> <output_text_file>")
        sys.exit(1)

    input_audio_file = sys.argv[1]
    output_text_file = sys.argv[2]

    # 1. Load the processor
    processor = AutoProcessor.from_pretrained("openai/whisper-medium.en")

    # 2. Create a distinct pad token if it doesn't exist
    #    (If your tokenizer already has a dedicated pad token, you can skip this step.)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<PAD>"})


    # 3. Load the Whisper model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en", low_cpu_mem_usage=True)

    # 4. Resize embeddings to account for new special tokens
    model.resize_token_embeddings(len(processor.tokenizer))

    # 5. Create the ASR pipeline with the updated tokenizer and model
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps=True,
        device="mps"
    )

    # Transcribe the audio
    result = transcriber(input_audio_file, return_timestamps=True)

    # Write the timestamped transcription to the output file
    with open(output_text_file, "w", encoding="utf-8") as f:
        for chunk in result["chunks"]:
            start_time, end_time = chunk["timestamp"]  # [start, end] in seconds
            text = chunk["text"]
            line = f"{start_time:.2f}s - {end_time:.2f}s: {text}"
            # Print to console  
            print(line) 
            # update the line to be just the text
            line = f"{text}"
            f.write(line + "\n")   # Write to file

    print(f"Transcription with timestamps saved to: {output_text_file}")

if __name__ == "__main__":
    # F=xxx python $F cdn/"${${F##*/}%.*}.mp3"
    main()