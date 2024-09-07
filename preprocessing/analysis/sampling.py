import os
import logging
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
logger = logging.getLogger(__name__)

# npz 파일을 처리하는 함수
def upsample_npz_file(file_path, target_rate):
    # 업샘플링 된 데이터 저장
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    if os.path.exists(output_file_path):
        return
    # npz 파일 불러오기
    data = np.load(file_path)
    upsampled_data = {}
    # 파일 내 배열 처리
    for key in data.files:
        signal = data[key]
        # 배열의 길이를 통해 현재 샘플링 주파수 계산 (1분 단위로 저장되어 있으니)
        current_sampling_rate = len(signal)
        
        # 업샘플링할 샘플 수 계산
        num_samples_target = int(target_rate * 60)  # 512 Hz로 1분 동안의 샘플 개수
        # 업샘플링
        upsampled_signal = resample(signal, num_samples_target)
        
        # 새로운 배열 저장
        upsampled_data[key] = upsampled_signal
    
    # np.savez(output_file_path, **upsampled_data)
    np.savez_compressed(output_file_path, **upsampled_data)


if __name__ == '__main__':
    # npz 파일을 읽어 들이고, 업샘플링 후 다시 저장할 폴더 경로
    input_folder = 'E:\\Capstone\\results\\npz'
    output_folder = 'E:\\Capstone\\results\\upsampled'
    target_sampling_rate = 100  # 100Hz로 샘플링

    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 입력 폴더의 모든 npz 파일 처리
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.npz'):
            file_path = os.path.join(input_folder, filename)
            try:
                upsample_npz_file(file_path, target_sampling_rate)
            except:
                pass

    print("샘플링 완료!")
