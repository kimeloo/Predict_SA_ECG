import os
import logging
import numpy as np
from collections import deque
from tools.get_apnea import get_apnea

logger = logging.getLogger(__name__)

def save(signal, file_path, s_freq):
    file_name = file_path.split('\\')[-1][:-4]
    end = len(signal)
    
    ann_path = file_path.replace('edfs', 'annotations-events-nsrr')
    ann_file = ann_path[:-4] + '-nsrr.xml'
    logger.info('Getting annotations for : {}'.format(ann_file))
    annotations = get_apnea(ann_file)
    annotations.reverse()
    try:
        annotation = annotations.pop()
    except:
        annotation = (None, None, None)
    
    step = s_freq * 60
    saved_signals = {}
    for sec in range(0, end, step):
        apnea = 'normal'
        apnea_start, apnea_end, _ = annotation
        if apnea_start == None:
            apnea = 'normal'
        elif int(apnea_end) < (sec//s_freq):
            if annotations:
                annotation = annotations.pop()
        if (((sec+1)//s_freq) <= int(apnea_end)) and (int(apnea_start) <= ((sec+1+step)//s_freq)):
            apnea = 'apnea'

        current_start = str((sec+1)//step).zfill(4)
        # current_end = str((sec+1+step)//step).zfill(4)
        signal_key = f'{current_start}_{apnea}'
        saved_signals[signal_key] = signal[sec:sec+step]

    os.makedirs(f'results/npz', exist_ok=True)
    np.savez_compressed(f'results/npz/{file_name}',**saved_signals)
    logger.info('Saved npz for : {}'.format(file_name))
