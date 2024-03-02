import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np

file="blues.00000.wav"

#Display time domain waveform
signal,sr=librosa.load(file,sr=22050)
librosa.display.waveshow(signal, sr=sr, color='pink')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Signal in time domain")
plt.show()

#fft spectrum
fft=np.fft.fft(signal)
magnitude=np.abs(fft)
frequency=np.linspace(0,sr,len(magnitude))
plt.plot(frequency,magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Full fft signal")
plt.show()

#In the above spectrum we have a mirror image spectrum output where only one side frequncy is important to display signal
left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(frequency)/2)]
plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Left fft signal")
plt.show()

#The above signal gives static information about signal and to know dynamic imformation we perform stft_>spectrogram
#It gives information of signal in both frequency and time domain
n_fft=2048 #number of samples basically the window to perform single ft
hop_length=512 #amount we shift to perform next ft

stft=librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
spectrogram=np.abs(stft)
librosa.display.specshow(spectrogram,sr=sr,hop_length=hop_length)
plt.colorbar(format='%+2.0f dB') 
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("spectrogram")
plt.show()

#log spectrogram
log_spectrogram=librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
plt.colorbar(format='%+2.0f dB') 
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("log spectrogram")
plt.show()

#MFCC
mfcc=librosa.feature.mfcc(y=signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
librosa.display.specshow(mfcc,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("mfcc")
plt.title("mfcc")
plt.show()










