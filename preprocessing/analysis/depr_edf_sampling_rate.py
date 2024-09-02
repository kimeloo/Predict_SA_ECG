import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_sampling(raw):
    return raw.info['sfreq']

# def get_channels(raw):
#     return raw.info['ch_names']

def sampling_rate(raw):
    samp_rate = get_sampling(raw)
    return samp_rate

def run():
    from tools.get_raw import get_raw
    files, data = get_raw(sampling_rate)
    data_dict = {'file':files, 's_rate':data}
    df = pd.DataFrame(data_dict)
    df.to_csv('results/sampling_rate.csv', index=False)

if __name__ == '__main__':
    logger = logging.getLogger()
    run()