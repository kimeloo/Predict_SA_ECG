import os
import numpy as np

def data_generator(npz_files, batch_size):
    while True:
        batch_data = []
        for npz_file in npz_files:
            try:
                # .npz 파일 열기
                data = np.load(os.path.join(npz_folder, npz_file))
                for array_name in data.files:
                    inputs = data[array_name]

                    if 'apnea' in array_name:
                        labels = 1
                    else:
                        labels = 0

                batch_data.append((inputs, labels))
            except:
                print(f"파일 {npz_file} 처리 중 오류 발생: 이 파일을 건너뜁니다.")
                continue

            # 배치 크기에 도달하면 yield
            if len(batch_data) == batch_size:
                yield np.array([x[0] for x in batch_data]), np.array([x[1] for x in batch_data])
                batch_data = []

        # 모든 파일을 다 돌았으면 다시 처음으로
        if len(batch_data) > 0:
            yield np.array([x[0] for x in batch_data]), np.array([x[1] for x in batch_data])

import tensorflow as tf
import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense

npz_folder = 'E:\\Capstone\\results\\upsampled'
npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
batch_size = 16

# RNN 모델 정의 (예시)
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(600, 1), return_sequences=False),  # LSTM 레이어
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # 이진 분류 (apnea/normal)
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습 (generator 사용)
# model.fit_generator(data_generator(npz_files, batch_size=16), steps_per_epoch=len(npz_files)//32, epochs=10)
model.fit(data_generator(npz_files, batch_size=batch_size), steps_per_epoch=len(npz_files)//batch_size, epochs=10)
