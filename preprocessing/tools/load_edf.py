import mne
import random
import platform
from pathlib import Path

class Edf_loader():
    def __init__(self, path):
        self.path = path
        self.load_edf()
    
    def get_raw(self):
        return self.raw

    def load_edf(self):
        self.raw = mne.io.read_raw_edf(self.path, preload=False, verbose="CRITICAL")
    
class Path_mgr():
    '''
    # init params
    ### operating_system
    "MacOS", "Windows"
    '''
    def __init__(self):
        self.operating_system = platform.system()
    
    def get_files(self, dataset, subset="", n_files=1, random_seed=42, all=False):
        '''
        # get_files() params
        ### dataset
        "bestair", "cfs", "hchs", "shhs"
        ### subset (default "")
        #### (for bestair)
        "baseline", "followup", "nonrandomized"
        #### (for shhs)
        "shhs1", "shhs2"
        ### n_files (default 1)
        1 <= n_files <= num_of_files
        ### random_seed (default 42)
        ### all (default False)
        if all is True, returns all files
        '''
        path = self.__get_path(dataset, subset)
        files = self.__fetch_all(path)
        if all:
            return files
        else:
            random.seed(random_seed)
            return random.sample(files, n_files)

    def __get_path_base(self):
        if self.operating_system == "Windows":
            base = 'E:/Capstone'
        elif self.operating_system == "Darwin":
            base = '/Volumes/eloo.iptime.org/Main/Capstone'
        else:
            raise Exception("Run in Windows or MacOS")
        return base

    def __get_path(self, dataset, subset):
        if subset:
            subset = subset+"/"
        base = self.__get_path_base()
        path = f'{base}/{dataset}/polysomnography/edfs/{subset}'
        return path

    def __fetch_all(self, root):
        path = Path(root)
        files = [str(file.resolve()) for file in path.rglob('*') if file.is_file()]
        return files