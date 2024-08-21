import logging
import numpy as np
from tqdm import tqdm
import mne

logger = logging.getLogger(__name__)

def is_valid_channel(raw, channel):
    if channel in raw.info['ch_names']:
        return True
    else:
        return False

def get_raw(func, channel="", return_raw=False):
    from tools.load_edf import Edf_loader, Path_mgr
    # datasets = ['bestair', 'cfs', 'shhs']
    datasets = ['bestair', 'shhs']
    subsets = {'bestair':['baseline', 'followup', 'nonrandomized'], 'shhs':['shhs1', 'shhs2']}

    tot_files = []
    for dataset in datasets:
        if dataset in subsets:
            subset = subsets[dataset]
        else:
            subset = ['']

        for s_set in subset:
            files = Path_mgr().get_files(
                dataset=dataset,
                subset=s_set,
                # n_files=1,
                # random_seed=42,
                all=True
                )
            
            tot_files.extend(files)
    print(f'files : {len(tot_files)}')
    results = []
    results_raw = []
    for file in tqdm(tot_files):
        logger.info(f'getting file : {file}')
        raw = Edf_loader(file).get_raw()
        if channel:
            ecg_channels = ["ecg", "ekg", "ecg1", "ecg2"]
            ecg_channels.extend(["ECG", "EKG", "ECG1", "ECG2"])
            if channel in ecg_channels:
                for ch_name in ecg_channels:
                    if is_valid_channel(raw, ch_name):
                        channel = ch_name
                        break
                else:
                    tot_files.remove(file)
                    logger.error('No ECG CHANNEL, remove : {}'.format(file))
                    continue
            elif is_valid_channel(raw, channel):
                pass
            else:
                logger.critical("INVALID CHANNEL NAME")
                return tot_files, [], []
            
            ecg_picks = mne.pick_channels(raw.info["ch_names"], include=[channel])
            raw = raw.copy().pick(picks=ecg_picks)

        
        results.append(func(raw))
        if return_raw:
            results_raw.append(raw)
    if return_raw:
        return tot_files, results, results_raw
    else:
        return tot_files, results

if __name__ == '__main__':
    print(f'Use this function by run(Your_Function), Returns result')