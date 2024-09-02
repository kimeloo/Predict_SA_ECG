import logging
import mne
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_raw(func, return_raw=False):
    from tools.load_edf import Edf_loader, Path_mgr
    from tools.save_np import save
    datasets = ['bestair', 'cfs', 'shhs']
    subsets = {'bestair':['baseline', 'followup', 'nonrandomized'], 'shhs':['shhs1', 'shhs2']}

    tot_files = []
    for dataset in datasets:
        subset = subsets[dataset] if dataset in subsets else ['']
        
        for s_set in subset:
            files = Path_mgr().get_files(
                dataset=dataset,
                subset=s_set,
                # n_files=20,
                # random_seed=42,
                all=True
                )
            
            tot_files.extend(files)
    print(f'files : {len(tot_files)}')
    files = []
    results = []
    results_raw = []
    freq = []
    ecg_channels = ["ecg", "ekg", "ecg1", "ecg2"]
    ecg_channels.extend(["ECG", "EKG", "ECG1", "ECG2"])
    for file in tqdm(tot_files):
        logger.info(f'getting file : {file}')
        raw = Edf_loader(file).get_raw()
        if raw[0] == None:
            logger.error('Skip File : {}'.format(file))
            continue
        elif raw[0] == 'mne':
            raw = raw[1]
            for ch_name in ecg_channels:
                if ch_name in raw.info['ch_names']:
                    break
            else:
                logger.error('No ECG CHANNEL, remove : {}'.format(file))
                continue
            s_freq = int(raw.info['sfreq'])
            ecg_picks = mne.pick_channels(raw.info["ch_names"], include=[ch_name])
            raw = raw.copy().pick(picks=ecg_picks)
            try:
                ecg_raw = raw.get_data()
            except AssertionError:
                logger.error("MNE Assertion error while opening, remove : {}".format(file))
                continue
        else:
            raw = raw[1]
            for ecg_idx, sig in enumerate(raw[1]):
                if sig['label'] in ecg_channels:
                    s_freq = int(sig['sample_rate'])
                    ecg_raw = raw[0][ecg_idx]
                    break
            else:
                logger.error('No ECG CHANNEL, remove : {}'.format(file))
                continue

        results.append(func(ecg_raw))
        files.append(file)
        freq.append(s_freq)
        try:
            save(ecg_raw, file, s_freq)
        except:
            logger.error('Numpy Save Error : {}'.format(file))
        if return_raw:
            results_raw.append(ecg_raw)
    if return_raw:
        return files, results, freq, results_raw
    else:
        return files, results, freq

if __name__ == '__main__':
    print(f'Use this function by run(Your_Function), Returns result')