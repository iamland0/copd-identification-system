import numpy as np
import matplotlib.pyplot as plt
import pyaudio


class KlasifikasiSuara(object):
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()


    def mainloop(self):
        while (self.stream.is_active()):
            self.plot()

    def plot(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(0, 2 * self.CHUNK, 2)
        ax.set_ylim(-10000, 10000)
        ax.set_xlim(0, self.CHUNK)  # x axis sejumlah chunk
        ax.set_title("Suara Paru-Paru")
        line, = ax.plot(x, np.random.rand(self.CHUNK))

        while True:
            data = self.stream.read(self.CHUNK)
            data = np.frombuffer(data, np.int16)
            line.set_ydata(data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

suara = KlasifikasiSuara()
suara.start()     # memulai mic
suara.mainloop()  # proses klasifikasi
suara.stop()
