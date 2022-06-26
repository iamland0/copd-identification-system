import json
import os
import math
import librosa

DATASET_PATH = "Respiratory_Sound_Database/audio"
JSON_PATH = "data_5.json"
SAMPLE_RATE = 22050
DURASI_DATASET = 20  # dalam detik
SAMPLES_PER_DATASET = SAMPLE_RATE * DURASI_DATASET


def ekstrak_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # simpan kelas, labels, mfcc pada dictionary
    data = {
        "kelas": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_DATASET / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop di semua folder kelas
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # memastikan yg diproses adalah subfolder dari path dataset
        if dirpath is not dataset_path:

            # simpan nama kelas
            semantic_label = os.path.split(dirpath)[-1]
            data["kelas"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # memproses semua file suara
            for f in filenames:

                # load file suara
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # memproses semua segmen pada file suara
                for d in range(num_segments):

                    # menghitung awal dan akhir sampel pada setiap segmen
                    awal = samples_per_segment * d
                    akhir = awal + samples_per_segment

                    # ekstrak mfcc
                    mfcc = librosa.feature.mfcc(signal[awal:akhir], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # simpan mfcc yang panjangnya sesuai dengan yang diharapkan
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # simpan hasil ekstraksi ciri ke file json
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    ekstrak_mfcc(DATASET_PATH, JSON_PATH, num_segments=5)
