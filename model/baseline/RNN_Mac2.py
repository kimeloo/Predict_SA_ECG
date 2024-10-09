import os
import numpy as np
import ftplib
from io import BytesIO
import keras

# SFTP 설정
from . import env
ftp_host, ftp_port, ftp_user, ftp_password = env.ftp_host, env.ftp_port, env.ftp_user, env.ftp_password
npz_folder = f'/ftp/HDD_Main/Capstone/results/upsampled'  # 서버 내의 경로

# FTP 접속 및 npz 파일 목록 불러오기
ftp = ftplib.FTP()
ftp.connect(host=ftp_host, port=ftp_port)
ftp.login(user=ftp_user, passwd=ftp_password)
npz_files = [f for f in ftp.nlst(npz_folder) if f.endswith('.npz')]

# Sequence 클래스를 이용한 데이터 생성기
class DataGenerator(keras.utils.Sequence):
    def __init__(self, npz_files, batch_size):
        self.npz_files = npz_files
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.npz_files) // self.batch_size

    def __getitem__(self, index):
        batch_files = self.npz_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = []

        for npz_file in batch_files:
            try:
                # SFTP에서 npz 파일을 메모리로 스트리밍
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

        # np.array로 변환
        inputs_batch = np.array([x[0] for x in batch_data])
        labels_batch = np.array([x[1] for x in batch_data])
        return inputs_batch, labels_batch

    def on_epoch_end(self):
        np.random.shuffle(self.npz_files)


# 배치 크기
batch_size = 16
epoch = 10

# RNN 모델 정의 (예시)
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(600, 1), return_sequences=False),  # LSTM 레이어
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # 이진 분류 (apnea/normal)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DataGenerator 인스턴스 생성
data_gen = DataGenerator(npz_files, batch_size=batch_size)

# 모델 학습
history = model.fit(data_gen, steps_per_epoch=len(npz_files)//batch_size, epochs=epoch)


# 모델 평가
test_loss, test_acc = model.evaluate(DataGenerator(npz_files, batch_size=batch_size), steps=len(npz_files)//batch_size)
print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

# 정확도, 오차, confusionmatrix 그리기
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
epoch_range = np.arange(1, epoch + 1)
# 정확도 그래프
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
# 오차 그래프
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
# # confusion matrix
# plt.subplot(1,3,3)
# conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
# plt.imshow(conf_mat_norm, cmap='Blues', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='red')


# FTP 연결 닫기
ftp.close()

plt.show()