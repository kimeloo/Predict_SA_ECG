import numpy as np

file_name = 'bestair-baseline-400001.npz'

# npz 파일 로드
loaded_data = np.load(f'results/npz/{file_name}')

# 저장된 segment_key들을 모두 확인 (current_start, current_end, apnea를 포함)
segment_keys = list(loaded_data.keys())
print(f"Available segment keys: ")
print(*segment_keys, sep="\n")

# 특정 segment_key로 시그널을 불러오는 방법
# specific_key = f'{current_start}_{current_end}_{apnea}'
# if specific_key in loaded_data:
#     signal_segment = loaded_data[specific_key]
#     print(f"Loaded signal for {specific_key}: {signal_segment}")
# else:
#     print(f"{specific_key} not found in the npz file.")
