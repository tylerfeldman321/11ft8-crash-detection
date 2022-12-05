import moviepy.editor as mpe
import matplotlib.pyplot as plt
import numpy as np
import wave
import os
from pathlib import Path

TEMP_DIR = 'temp'


def process_audio(files: str, output_fps=15, visualize=False):
    with wave.open(files, 'r') as wav_file:
        graphsig = wav_file.readframes(-1)  # -1 indicates all or max frames
        graphsig = np.frombuffer(graphsig, dtype="int16")
        f_rate = wav_file.getframerate()  # Get the frame rate
        output_rate = f_rate / output_fps

        # Split the data into channels
        channels = []
        n_channels = wav_file.getnchannels()
        for channel in range(wav_file.getnchannels()):
            channels.append(graphsig[channel::n_channels])
            pad_length = int(output_rate - (len(channels[channel]) % output_rate))
            padding = np.zeros(pad_length)
            channels[channel] = np.concatenate([channels[channel], padding])

        num_samples = int(np.ceil(len(graphsig)/len(channels)/f_rate*output_fps))
        time = np.linspace(0,
                           len(graphsig)/len(channels)/f_rate,
                           num=num_samples)
        avg_signal = np.mean(np.abs(np.array(channels)), axis=0)
        avg_signal_framewise = np.max(avg_signal.reshape(-1, int(output_rate)), axis=1)
        if visualize:
            plot_signal(time, avg_signal_framewise, files)
    return avg_signal_framewise


def plot_signal(time, signal, fname):
    plt.clf()
    plt.figure(1)
    plt.plot(time, signal)
    plt.title('Sound Wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    plt.savefig(fname + ".png")


def generate_data(video_file, file_name):
    if not video_file.endswith('mp4'):
        raise Exception('Provided file is not in mp4 format.')
    video = mpe.VideoFileClip(video_file)
    wav_file = os.path.join(TEMP_DIR, file_name + '.wav')
    video.audio.write_audiofile(wav_file)
    return process_audio(wav_file)


def main():
    directory = "data/crash samples/2021-07-12_bump-c169"

    files = Path(directory).glob('*')
    count = 0
    for file in files:
        count += 1
        print(count)
        if str(file).endswith('mp4'):
            print(file)
            video = mpe.VideoFileClip(str(file))
            wav_file = str(file) + ".wav"
            print(f'Video is {video.audio.duration} seconds long.')
            video.audio.write_audiofile(wav_file)
            process_audio(wav_file, visualize=True)


if __name__ == '__main__':
    main()
