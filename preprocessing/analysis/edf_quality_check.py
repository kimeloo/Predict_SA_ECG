import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

def quality_check(raw):
    missing_values = np.isnan(raw).sum()
    return missing_values

def run():
    from tools.get_raw import get_raw
    files, data, freq = get_raw(quality_check, return_raw=False)
    result = {'file':files, 'missing':data, 'freq':freq}
    df = pd.DataFrame(result)
    df.to_csv('results/quality_check.csv', index=False)

if __name__ == '__main__':
    logger = logging.getLogger()
    run()