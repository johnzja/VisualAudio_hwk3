import pyaudio
import wave



CHUNK = 1024
wf = wave.open('audio.wav', 'rb')
data = wf.readframes(CHUNK)
p = pyaudio.PyAudio()

FORMAT = p.get_format_from_width(wf.getsampwidth())
CHANNELS = wf.getnchannels()
RATE = wf.getframerate()

print('FORMAT: {} \nCHANNELS: {} \nRATE: {}'.format(FORMAT, CHANNELS, RATE))

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=CHUNK,
                output=True)
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK)
