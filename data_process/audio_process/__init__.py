import os
# import ffmpeg  # pip install ffmpeg-python

def extract_wav_from_video(video_file, wav_file, overwrite_exist:bool=False):
    if overwrite_exist or not os.path.exists(wav_file):
        dirname = os.path.dirname(wav_file)
        os.makedirs(dirname, exist_ok=True)

        extract_wav_cmd = f'ffmpeg -i {video_file} -f wav -ar 16000 {wav_file}'
        os.system(extract_wav_cmd)

        # # Open the MP4 file using the ffmpeg.input() function
        # mp4_file = ffmpeg.input(video_file)
        # # Get the WAV data from the file using the ffmpeg.output() function
        # wav_data, _ = ffmpeg.output(mp4_file, f="wav").run()
        print(f'audio file saved to {wav_file}')
    else:
        print(f'skip, audio file {wav_file} exist')