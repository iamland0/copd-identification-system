import numpy as np
import pyaudio
import librosa
from tensorflow.keras.models import load_model
import wave
import time

class KlasifikasiSuara(object):
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK = 1024 * 2
        self.DURASI_REKAM = 5
        self.DURASI = 4
        self.SAMPLE_DURASI = self.RATE * self.DURASI
        self.SAMPLE_DURASI_INT = int(self.RATE * self.DURASI)
        self.p = None
        self.stream = None
        self.savedModel = load_model('models/ppokmodelh5.h5', compile=False)

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

    def simpanWav(self):
        frames = []
        print("Mendengarkan Suara Paru-Paru...")
        for i in range(0, int(self.RATE/self.CHUNK * self.DURASI_REKAM)): #rekam 5 detik
            data = self.stream.read(self.CHUNK)
            frames.append(data)

        print("Memproses Suara Paru-Paru...")
        file = wave.open("deteksiwav/deteksi.wav", "wb") # membuka file dengan mode 'write bytes'
        file.setnchannels(self.CHANNELS)
        file.setsampwidth(self.p.get_sample_size(self.FORMAT))
        file.setframerate(self.RATE)
        file.writeframes(b''.join(frames)) # merubah frames ke bytes
        file.close()

    def ekstraksiCiri(self,  file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load file suara
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= self.SAMPLE_DURASI_INT:
            # ambil 4 detik awal
            signal = signal[:self.SAMPLE_DURASI_INT]
            mfccs = librosa.feature.mfcc(signal, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfccs = mfccs.T

            return mfccs


    def prediksi(self, file_path):

        start = time.time() #mulai timer untuk menghitung waktu komputasi

        # ekstrak MFCC
        MFCCs = self.ekstraksiCiri(file_path)

        # menambah dimensi pada array input data untuk sample - model.predict() membutuhkan 4D array
        MFCCs = MFCCs[np.newaxis, ...] # array shape (1, 173, 13, 1)

        # melakukan prediksi
        predictions = self.savedModel.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        print("Hasil Diagnosis: ")

        if predicted_index == 0:
            hasil = print("Suara Paru-Paru Terindikasi Penyakit Paru Obstruktif Kronik (PPOK)")
        elif predicted_index == 1:
            hasil = print("Suara Paru-Paru Sehat")

        end = time.time() #stop timer
        print(f"Waktu Komputasi : {end - start}\n")

        return hasil


    def mainloop(self):
        while (self.stream.is_active()): #matikan baris ini dan 2 baris kebawah jika klasifikasi dari file
            self.simpanWav()
            self.prediksi("deteksiwav/deteksi.wav")
        #self.prediksi("ppok_test/147_1b3_Tc_mc_AKGC417L.wav") #klasifikasi dari file



suara = KlasifikasiSuara()
suara.start()     # memulai mic
suara.mainloop()  # proses klasifikasi
suara.stop()