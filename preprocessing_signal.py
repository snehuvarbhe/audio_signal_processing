import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np

file="blues.00000.wav"

# Display time domain waveform

signal,sr=librosa.load(file,sr=22050)
librosa.display.waveshow(signal, sr=sr, color='pink')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#fft spectrum
fft=np.fft.fft(signal)
magnitude=np.abs(fft)
frequency=np.linspace(0,sr,len(magnitude))
plt.plot(frequency,magnitude)
plt.show()