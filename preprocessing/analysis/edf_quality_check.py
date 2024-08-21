import logging
import matplotlib.pyplot as plt
import pandas as pd
logger = logging.getLogger(__name__)

def quality_check(raw):
    try:
        nd_raw = raw.get_data()
    except AssertionError:
        logger.error("get_data AssertionError")
        return 999

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
    logger = logging.getLogger()
    run()