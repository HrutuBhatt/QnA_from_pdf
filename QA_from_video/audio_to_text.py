import yt_dlp

def download(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',  # Save as title.mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

video_url = "https://youtu.be/6ChzCaljcaI?si=1cLrMpAVvulipn7f"
download(video_url)

import os

def split_video(input_file="video.mp3", chunk_length=15*60):  # 40 minutes
    os.makedirs("chunks", exist_ok=True)

    # Print audio duration
    print("Checking duration...")
    os.system(f'ffmpeg -i "{input_file}" 2>&1 | grep "Duration"')

    # Split the file
    print("Splitting audio...")
    os.system(f'ffmpeg -i "{input_file}" -f segment -segment_time {chunk_length} -c copy chunks/output%03d.mp3')

split_video("video.mp3")

import os
from groq import Groq
from glob import glob

client = Groq(api_key="API_KEY")

chunk_folder = "chunks"
chunk_files = sorted(glob(os.path.join(chunk_folder, "*.mp3")))

full_script = ""
for idx, filepath in enumerate(chunk_files):
    filename = os.path.basename(filepath)
    print(f"Transcribing chunk {idx+1}/{len(chunk_files)} : {filename}")
    with open(filepath, "rb") as file:
      transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
      )
      full_script += transcription.text + "\n"

print(chunk_files)

print(full_script)


with open("final_transcript.txt", "w", encoding="utf-8") as f:
    f.write(full_script)

