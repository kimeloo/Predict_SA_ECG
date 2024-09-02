from pyedflib import highlevel
import os
import numpy as np
from glob import glob
from tools.load_edf import Path_mgr

# datasets = ['bestair', 'shhs', 'cfs']
datasets = ['shhs']
# subsets = {'bestair':['baseline', 'followup', 'nonrandomized'], 'shhs':['shhs1', 'shhs2']}
subsets = {'shhs':['shhs1']}

tot_files = []
for dataset in datasets:
    subset = subsets[dataset] if dataset in subsets else ['']

    for s_set in subset:
        files = Path_mgr().get_files(
            dataset=dataset,
            subset=s_set,
            n_files=1,
            # random_seed=42,
            # all=True
            )
        
        tot_files.extend(files)

ecg_list = ['ecg', 'ekg', 'ecg1', 'ecg2', 'ECG', 'EKG', 'ECG1', 'ECG2']

for file in tot_files:
    edf = highlevel.read_edf(file)
    file_name = file.split('\\')[-1][:-4]
    
    ann_path = file.replace('edfs', 'annotations-events-nsrr')
    ann_file = ann_path[:-4] + '-nsrr.xml'

    
    
    # print(edf)
    # print(*edf[1][0], sep="\n")
    # break

    print("\n"+file)
    for ecg_idx, sig in enumerate(edf[1]):
        if sig['label'] in ecg_list:
            for key, val in zip(list(sig.keys()), list(sig.values())):
                print(f'{key} : {val}')
            break
    print(len(edf[0][ecg_idx]))
    print(type(edf[0][ecg_idx]))
    
    end = len(edf[0][ecg_idx])
    s_freq = int(edf[1][ecg_idx]['sample_rate'])
    signal = edf[0][ecg_idx]
    
    for sec in range(0, end, s_freq):
        current_start = str(sec).zfill(9)
        current_end = str(sec+s_freq).zfill(9)
        os.makedirs(f'results/{file_name}', exist_ok=True)
        np.save(f'results/{file_name}/{current_start}_{current_end}',signal[sec:sec+s_freq])