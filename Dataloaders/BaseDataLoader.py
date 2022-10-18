import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import Dataset
import os
import torch
from colorednoise import powerlaw_psd_gaussian as ppg
from scipy.signal import butter, filtfilt
from biosppy.signals.ecg import christov_segmenter

class BaseECGLoader(Dataset):

    def __init__(self, datapoints: int, samples: list, snr_db: int, noise_color: int = 0):

        super(BaseECGLoader, self).__init__()

        self.datapoints = datapoints

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        # Load dataset
        self.dataset, self.fs = self.load_data(samples)

        # Get dataset dimensions
        self.samples, self.signal_length, self.num_channels = self.dataset.shape

        # Add gaussian white noise
        self.observations = self.add_noise(self.dataset, snr_db, noise_color)

        # Get QRS-peak labels
        print('--- Centering Heartbeats ---')
        self.labels = self.find_peaks(self.observations)

        # Center data
        self.centered_observations, self.centered_states, self.overlaps = self.center(self.observations,
                                                                                      self.dataset,
                                                                                      datapoints,
                                                                                      self.labels)



    def load_data(self, samples: list) -> (torch.Tensor, int):
        """
        Load the dataset as a tensor with dimensions: (Samples, Time, channel)
        :param samples: Array of samples to choose from
        :return: Raw dataset and sampling frequency
        """
        raise NotImplementedError

    def add_noise(self, dataset: torch.Tensor, snr_db: int, noise_color: int) -> torch.Tensor:
        """
        Add noise of a specified color and snr
        :param snr_db: Signal to noise ratio  in decibel
        :param noise_color: Color of noise 0: white, 1: pink, 2: brown, ...
        :return: Tensor of noise data
        """
        # Calculate signal power along time domain
        signal_power_db = 10 * torch.log10(dataset.var(1) + dataset.mean(1) ** 2)

        # Calculate noise power
        noise_power_db = signal_power_db - snr_db
        noise_power = torch.pow(10, noise_power_db / 20)

        # Set for reproducibility
        random_state = 42

        # Generate noise
        noise = [ppg(noise_color, self.signal_length, random_state=random_state) for _ in range(self.num_channels)]
        noise = torch.tensor(np.array(noise)).T.float() * noise_power

        # Add noise
        noisy_data = self.dataset + noise

        return noisy_data

    def find_peaks(self, observations: torch.Tensor) -> list:
        """
        Find QRS peaks from observations
        :param observations: Tensor of observations
        :return: List of peak indices
        """
        # Create label list
        labels = []

        for sample in observations:

            # Get labels using a christov segmenter
            """Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, 
            BioMedical Engineering OnLine 2004, vol. 3:28, 2004 """
            sample_labels = christov_segmenter(sample[:, 0], sampling_rate=self.fs)[0]

            labels.append(sample_labels)

        return labels

    def center(self, observations: torch.Tensor, states: torch.Tensor, datapoints: int, labels: list) \
            -> (torch.Tensor, torch.Tensor, list):
        """
        Center observations and noiseless data with given labels and time horizon°
        :param observations: Tensor of noisy observations
        :param states: Tensor of noiseless states
        :param datapoints: Number of datapoints
        :param labels: Labels of QRS-peaks
        :return: Centered observation, centered states and a list overlaps
        """
        # Allocate data buffers
        centered_states = []
        centered_observations = []
        overlaps = []

        for n_sample, (obs_sample, state_sample, label) in enumerate(zip(observations, states, labels)):

            last_upper_index = 0

            # Create data buffers for the current sample
            sample_centered_observations = []
            sample_centered_states = []
            sample_overlaps = []

            for n_beat, beat in enumerate(label):

                # Get lower and upper indices
                lower_index = beat - int(datapoints / 2)
                upper_index = beat + int(datapoints / 2)

                # Ignore first and last detected beat, since we can not determine where they started/ended
                if lower_index < 0 or upper_index > self.signal_length:
                    last_upper_index = upper_index
                    continue

                else:

                    # Cut out data around QRS-peak
                    single_heartbeat_observation = obs_sample[lower_index: upper_index]
                    single_heartbeat_state = state_sample[lower_index: upper_index]

                    # Calculate the overlap for stiching everything back together
                    overlap = max(last_upper_index - lower_index, 0)

                    # Append to data buffers
                    sample_centered_observations.append(single_heartbeat_observation)
                    sample_centered_states.append(single_heartbeat_state)
                    sample_overlaps.append(overlap)

            # Append to data buffers
            centered_observations.append(torch.stack(sample_centered_observations))
            centered_states.append(torch.stack(sample_centered_states))
            overlaps.append(sample_overlaps)

            return torch.cat(centered_observations), torch.cat(centered_states), overlaps

    def __len__(self):
        return self.centered_states.shape[0]

    def __getitem__(self, item):
        return self.centered_observations[item], self.centered_states[item]
