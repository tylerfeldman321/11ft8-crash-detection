import moviepy.editor as mpe
import matplotlib.pyplot as plt
import numpy as np
import wave, sys

from pathlib import Path
from scipy.fftpack import fft

#find the place where there's the highest average sound over time?

def makegraph(files: str):

	# reading the audio file
	audiowrt = wave.open(files)
	
	# reads all the frames
	# -1 indicates all or max frames
	graphsig = audiowrt.readframes(-1)
	graphsig = np.frombuffer(graphsig, dtype ="int16")
	
	# gets the frame rate
	f_rate = audiowrt.getframerate()
	print(f_rate)

	print(len(graphsig))
	time = np.linspace(
		0, # start
		len(graphsig) / f_rate,
		num = len(graphsig)
	)
	#print(time)
	# using matplotlib to plot
	#print("figure print")
	plt.clf()
	plt.figure(1)
	
	# title of the plot
	plt.title("Sound Wave")
	
	# label of x-axis
	plt.xlabel("Time")
	
	# actual plotting
	plt.plot(time, graphsig)
	
	# shows the plot
	# in new window
	#plt.show()

	plt.savefig(files + ".png")

directory = "Project Videos"

files = Path(directory).glob('*')
count = 0
for file in files:
    count+=1
    print(count)
    str(file).endswith('mp4')
    print(file)
    video = mpe.VideoFileClip(str(file))
    wavfile = str(file) + ".wav"
    video.audio.write_audiofile(wavfile)
    makegraph(wavfile)


# shows the sound waves




