from Dataloaders.BaseDataLoader import BaseECGLoader
import torch
import numpy as np
import os
import glob
import wfdb


class MIT_BIH_DataLoader(BaseECGLoader):

    def __init__(self, datapoints: int, samples: list, snr_db: int, noise_color: int = 0):

        super(MIT_BIH_DataLoader, self).__init__(datapoints, samples, snr_db, noise_color)

    def load_data(self, samples: list) -> (torch.Tensor, int):
        """
        Download dataset from PhysioNet if it is not downloaded yet, and return the all sample patient recordings
        :param samples: List of all desired patient samples
        :return: (dataset, sampling frequency)
        """
        # Define sampling frequency
        samples_per_second = 360

        # Get location of this file
        file_location = os.path.dirname(os.path.realpath(__file__))

        # Get path to dataset location
        path_to_dataset = file_location + '\\..\\Datasets\\MIT-BIH-Arrhythmia_Database'

        # Check if the folder exist and if not, create it
        if not os.path.exists(path_to_dataset):
            os.mkdir(path_to_dataset)
            wfdb.io.dl_database('mitdb', path_to_dataset + '\\')

        # Get filenames from folder
        raw_data_files = glob.glob(path_to_dataset + '\\' + '*.hea')
        raw_data_files = [raw_data_files[index] for  index in samples]

        # Get filenames from folder
        raw_annotation_files = glob.glob(path_to_dataset + '\\' + '*.atr')
        raw_annotation_files = [raw_annotation_files[index] for index in samples]

        annotation_files = [wfdb.rdann(annotation_file[:-4], 'atr' ) for annotation_file in raw_annotation_files]

        # Read the data stream
        dataset = [wfdb.rdrecord(raw_data_file[:-4]).p_signal for raw_data_file in raw_data_files]
        self.mit_labels = [file.sample for file in annotation_files]

        # Convert to torch tensor
        dataset = torch.tensor(np.array(dataset)).float()

        return dataset, samples_per_second

    def find_peaks(self, observations: torch.Tensor) -> list:
        return self.mit_labels

    def __len__(self):
        return 42
