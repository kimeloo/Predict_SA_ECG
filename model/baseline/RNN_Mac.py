import os
import numpy as np
import ftplib
from io import BytesIO

# FTP 설정
from .env import ftp_host, ftp_port, ftp_user, ftp_password
npz_folder = '/Main/Capstone/results/upsampled'  # FTP 서버 내의 경로

# FTP 접속 및 npz 파일 목록 불러오기
ftp = ftplib.FTP(ftp_host, ftp_port)
ftp.login(ftp_user, ftp_password)
npz_files = [f for f in ftp.nlst(npz_folder) if f.endswith('.npz')]

def data_generator(npz_files, batch_size):
    while True:
        batch_data = []
        for npz_file in npz_files:
            try:
                # FTP에서 npz 파일을 메모리로 스트리밍
                data_stream = BytesIO()
                ftp.retrbinary(f'RETR {os.path.join(npz_folder, npz_file)}', data_stream.write)
                data_stream.seek(0)
                
                # .npz 파일 열기
                data = np.load(data_stream)
                for array_name in data.files:
                    inputs = data[array_name]

                    if 'apnea' in array_name:
                        labels = 1
                    else:
                        labels = 0

                batch_data.append((inputs, labels))

                # 메모리 해제
                data_stream.close()
                data.close()

            except Exception as e:
                print(f"파일 {npz_file} 처리 중 오류 발생: {e}")
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
model.fit(data_generator(npz_files, batch_size=batch_size), steps_per_epoch=len(npz_files)//batch_size, epochs=10)

# FTP 연결 닫기
ftp.quit()
