import whisperx
import gc 
import time

device = "cuda" 
audio_file = "/mnt/audio/en_30_test.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model_dir = "/mnt/models"

# 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
load_model_s = time.time()
model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir, language="en", task="transcribe")
load_model_e = time.time()

inference_s = time.time()
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment
whisperx_result = result["segments"]
inference_e = time.time()

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
Align_load_s = time.time()
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir=model_dir)
Align_load_e = time.time()
Align_s = time.time()
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
Align_whisperx_result = result["segments"]
Align_e = time.time()

# print(result["segments"]) # after alignment


# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
Assign_speaker_load_s = time.time()
diarize_model = whisperx.DiarizationPipeline(use_auth_token="XXX", device=device)
Assign_speaker_load_e = time.time()

# add min/max number of speakers if known
diarize_segments_s = time.time()
diarize_segments = diarize_model(audio)
diarize_segments_e = time.time()

# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
assign_word_speakers_s = time.time()
result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs
assign_word_speakers_result = result["segments"]
assign_word_speakers_e = time.time()

print("====="*20)
print("load whisper: ", load_model_e - load_model_s)
print("whisper inference: ", inference_e - inference_s)
combined_text = ''.join(entry['text'] for entry in whisperx_result)  
print("raw output:", combined_text)
print("====="*20)

print("load Align: ", Align_load_e - Align_load_s)
print("Align inference: ", Align_e - Align_s)
print("[ Start  ->    End ] Text")
for entry in Align_whisperx_result:
    print(f"[{float(entry['start']):06.3f}  ->  {float(entry['end']):06.3f}] {entry['text'].strip()}")
print("====="*20)

print("load Assign_speaker: ", Assign_speaker_load_e - Assign_speaker_load_s)
print("diarize segments: ", diarize_segments_e - diarize_segments_s)
print("assign_wordspeakers: ", assign_word_speakers_e - assign_word_speakers_s)
print(diarize_segments)
# print(assign_word_speakers_result)
print("====="*20)

# XXX = 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

