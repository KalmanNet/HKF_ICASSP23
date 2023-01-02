import glob
from scipy.io import loadmat
from Dataloaders.BaseDataLoader import BaseECGLoader
import torch
import os
import numpy as np


class ProprietaryDataLoader(BaseECGLoader):

    def __init__(self, datapoints: int, sample: int, snr_db: int, noise_color: int = 0):
        super(ProprietaryDataLoader, self).__init__(datapoints, sample, snr_db, noise_color)


    def load_data(self, sample: int) -> (torch.Tensor, int):

        sampling_rate = 500

        # Get location of this file
        file_location = os.path.dirname(os.path.realpath(__file__))

        # Get path to dataset location
        path_to_dataset = file_location + '\\..\\Datasets\\Proprietary_Database'

        # Check if the folder exist and if not, create it
        if not os.path.exists(path_to_dataset):
            print('Please check with: R.Vullings@tue.nl, to obtain dataset')
            raise SystemExit

        # Specify folder
        exact_path = '\\simulatedFromAdult_500Hz\\traindata\\'
        exact_path = path_to_dataset + exact_path

        # Get filenames from folder
        mat_files = glob.glob(exact_path + '*.mat')
        mat_files = [mat_files[sample] for sample in [sample]]

        # Read the data stream
        dataset = [loadmat(mat_file)['ECG'] for mat_file in mat_files]

        # Convert to torch tensor
        dataset = torch.tensor(np.array(dataset)).float()

        return dataset.mT, sampling_rate

