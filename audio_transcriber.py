import torch  # has pytorch(complex version of scikit learn) accelerate which uses gpu - could make it crash
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)  # nothing to do with model, from huggingface
from datasets import load_dataset

device = (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # if there is available gpu, set device to that, if not, use cpu, dependency injection
torch_dtype = (
    torch.float16 if torch.cuda.is_available() else torch.float32
)  # decides how much memory each float in the program will take, dependency injection

model_id = "openai/whisper-large-v3"  # tells it what model to use

# creates the model, passing in needed information
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)  # sets the device which the model will use

processor = AutoProcessor.from_pretrained(
    model_id
)  # comes from transformers library, tells it which model it will use

# creates a pipeline - primaryschool function machine
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset(
    "distil-whisper/librispeech_long", "clean", split="validation"
)  # load in the data you want to compute
sample = dataset[0]["audio"]

# result = pipe(sample) #performs the function on the data


# convert to mp3 in command line: .\ffmpeg.exe -i Recording.m4a -c:v copy -c:a libmp3lame -q:a 4 output.mp3
def transcribe_audio(mp3_file):
    result = pipe(mp3_file)

    return result["text"]  # outputs this
