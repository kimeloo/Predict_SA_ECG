import os
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, npz_folder, batch_size=32, max_len=100, shuffle=True):
        self.npz_folder = npz_folder
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.npz_files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_files = self.npz_files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.npz_files)

    def __data_generation(self, batch_files):
        X = []
        y = []

        for npz_file in batch_files:
            try:
                # 손상된 파일은 무시
                data = np.load(os.path.join(self.npz_folder, npz_file))
                for array_name in data.files:
                    array = data[array_name]
                    padded_array = tf.keras.preprocessing.sequence.pad_sequences([array], maxlen=self.max_len, padding='post')

                    if 'apnea' in array_name:
                        label = 1
                    else:
                        label = 0

                    X.append(padded_array[0])
                    y.append(label)
            
            except:
                print(f"파일 {npz_file} 처리 중 오류 발생: 이 파일을 건너뜁니다.")
                continue

        X = np.array(X)
        y = np.array(y)

        return X, y

# npz 파일이 저장된 폴더 경로 설정
npz_folder = 'E:\\Capstone\\results\\upsampled'

# 데이터 제너레이터 인스턴스 생성
batch_size = 512
train_generator = DataGenerator(npz_folder, batch_size=batch_size, max_len=600, shuffle=False)



# RNN 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(600, 1), return_sequences=False),  # LSTM 레이어
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류 (apnea/normal)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_generator, epochs=10)

# 모델 평가 (test generator 필요 시 별도 생성)
