from audio_transcriber import transcribe_audio
from text_summeriser import summarise_text
from pathlib import Path
import os
import os.path

def summarise_mp3(audio_file):
    file_name = os.path.splitext(audio_file)[0]
    text_file = file_name + "_transcript.txt"
    if not os.path.exists(text_file):
        audio_transcript = transcribe_audio(audio_file)
        with open(text_file, "w") as t: 
            t.write(audio_transcript)
    else:
        with open(text_file, "r") as t: 
            audio_transcript = t.read()
    
    summarised_text = summarise_text(audio_transcript)

    print(summarised_text)
    summary_text_file = file_name + "_summary.txt"
    with open(summary_text_file, "w") as t: 
            t.write(summarised_text[0]["summary_text"])

if __name__ == "__main__":
    file_to_transcribe = input("input the name of the mp3 file to be transcribed:  ")
    summarise_mp3(file_to_transcribe)
