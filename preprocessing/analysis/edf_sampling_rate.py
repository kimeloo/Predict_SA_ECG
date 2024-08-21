from tqdm import tqdm
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] : [%(name)15s] --- %(message)s")
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
log_warning_formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] : [%(name)15s] --- %(message)s")
log_warning_handler = logging.FileHandler("event.log")
log_warning_handler.setLevel(logging.INFO)
log_warning_handler.setFormatter(log_warning_formatter)
logger.addHandler(log_handler)
logger.addHandler(log_warning_handler)

def get_sampling(raw):
    return raw.info['sfreq']

def get_channels(raw):
    return raw.info['ch_names']

if __name__ == '__main__':
    from tools.load_edf import Edf_loader, Path_mgr

    datasets = ['bestair', 'cfs', 'shhs']
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
                # n_files=5,
                # random_seed=42,
                all=True
                )
            tot_files.extend(files)

    data = []
    for file in tqdm(tot_files):
        # print(file)
        raw = Edf_loader(file).get_raw()
        sampling_rate = get_sampling(raw)
        channels = get_channels(raw)
        data.append({'file':file, 's_rate': sampling_rate, 'channels': channels})
        # print(sampling_rate,end='\n\n')
    
    # print(*data, sep="\n")
    df = pd.DataFrame(data)
    df.to_csv('sampling_rate.csv', index=False)