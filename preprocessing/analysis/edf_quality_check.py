import logging
import matplotlib.pyplot as plt
import pandas as pd
logger = logging.getLogger(__name__)

def quality_check(raw):
    nd_raw = raw.get_data()
    # except AssertionError:
    #     logger.error("get data AssertionError")
    #     chunk_size = int(raw.info['sfreq'] * 1)
    #     data_chunks, times_chunks = get_data_in_chunks(raw=raw, chunk_size=chunk_size, picks=None, return_times=True)
    #     return 
    #     results.append(sum(func(chunk) for chunk in data_chunks))

    df = pd.DataFrame(nd_raw.T)
    missing_values = df.isnull().sum().sum()
    return missing_values

def run():
    from tools.get_raw import get_raw
    files, data = get_raw(quality_check, channel="ecg", return_raw=False)
    result = {'file':files, 'missing':data}
    df = pd.DataFrame(result)
    df.to_csv('results/quality_check.csv', index=False)

if __name__ == '__main__':
    run()