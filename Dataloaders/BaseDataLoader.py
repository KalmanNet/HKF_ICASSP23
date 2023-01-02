import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import Dataset
import os
import torch
from colorednoise import powerlaw_psd_gaussian as ppg
from scipy.signal import butter, lfilter, filtfilt
from biosppy.signals.ecg import christov_segmenter, engzee_segmenter
from biosppy.signals.ecg import *


class BaseECGLoader(Dataset):

    def __init__(self, datapoints: int, sample: int, snr_db: int or None, noise_color: float = 0,
                 dev: str= 'cpu'):

        super(BaseECGLoader, self).__init__()

        self.datapoints = datapoints

        self.file_location = os.path.dirname(os.path.realpath(__file__))

        # Load dataset
        self.dataset, self.fs = self.load_data(sample)

        b, a = butter(6, [0.5, 90], fs = self.fs, btype = 'bandpass')  # noqa

        self.dataset = torch.tensor(filtfilt(b, a, self.dataset, axis=1).copy()).float()

        # Get dataset dimensions
        self.samples, self.signal_length, self.num_channels = self.dataset.shape

        if snr_db is not None:
            # Add gaussian white noise
            self.observations = self.add_noise(self.dataset, snr_db, noise_color)
        else:
            self.observations = self.dataset

        # Get QRS-peak labels
        print('--- Centering Heartbeats ---')
        # self.labels = self.find_peaks(self.observations)
        self.labels = self.find_peaks(self.observations)

        # Center data
        self.centered_observations, self.centered_states, self.overlaps, self.start_indices, self.end_indices = self.center(
            self.observations,
            self.dataset,
            datapoints,
            self.labels)


    def load_data(self, samples: int) -> (torch.Tensor, int):
        """
        Load the dataset as a tensor with dimensions: (Samples, Time, channel)
        :param samples: Array of samples to choose from
        :return: Raw dataset and sampling frequency
        """
        raise NotImplementedError

    def add_noise(self, dataset: torch.Tensor, snr_db: int, noise_color: float) -> torch.Tensor:
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
        print(f'Noise power: {10*torch.log10(noise_power.mean()).item()}[dB]')
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



        b, a = butter(4, [0.5, 90], fs=self.fs, btype='bandpass')  # noqa

        filtered_obs = torch.tensor(filtfilt(b, a, observations, axis=1).copy()).float()

        from utils.MovingAverage import moving_average
        ma1 = moving_average(filtered_obs, window_size=11, axis=1)
        ma2 = moving_average(ma1, window_size=11, axis=1)

        # Create label list
        labels = []

        for sample in ma2:
            # Get labels using a christov segmenter
            """Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, 
            BioMedical Engineering OnLine 2004, vol. 3:28, 2004 """
            sample_labels = engzee_segmenter(sample[:, 0], sampling_rate=self.fs)[0]

            labels.append(sample_labels)

        return labels

    def check_fit(self):

        from matplotlib import pyplot as plt

        num_plot = 10_00
        plt.plot(self.dataset[0, :num_plot].squeeze())
        labels = self.labels[0][np.where(self.labels[0] < num_plot)]
        plt.vlines(labels, -1, 1, 'r')
        plt.show()

        offset = self.labels[0][0] - int(self.datapoints / 2)
        obs = self.dataset[0, offset:num_plot + offset].squeeze()
        from utils.Stich import stich
        stich_obs = stich(self.centered_observations, self.overlaps[0])
        plt.plot(obs)
        plt.show()
        plt.plot(stich_obs[:num_plot].squeeze())
        plt.show()

        1

    def center(self, observations: torch.Tensor, states: torch.Tensor, datapoints: int, labels: list) \
            -> (torch.Tensor, torch.Tensor, list, list):
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
        start_indices = []
        end_indices = []

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
                    # overlap = max(last_upper_index - lower_index, 0)
                    overlap = last_upper_index - lower_index

                    # Append to data buffers
                    sample_centered_observations.append(single_heartbeat_observation)
                    sample_centered_states.append(single_heartbeat_state)
                    sample_overlaps.append(overlap)
                    last_upper_index = upper_index

                    start_indices.append(lower_index)
                    end_indices.append(upper_index)

            # Append to data buffers
            centered_observations.append(torch.stack(sample_centered_observations))
            centered_states.append(torch.stack(sample_centered_states))
            overlaps.append(sample_overlaps)

        return torch.cat(centered_observations), torch.cat(centered_states), overlaps, start_indices, end_indices

    def __len__(self):
        return self.centered_states.shape[0]

    def __getitem__(self, item):
        return self.centered_observations[item], self.centered_states[item]

    def distort(self, lower_index: int, upper_index: int, time_index: int, typ: str, amplitude_mul: float = 1.5,
                resample_mul: float = 1.5):

        if typ == 'jump':

            self.centered_observations[time_index:, lower_index:upper_index] = \
                self.centered_observations[time_index:, lower_index:upper_index] + amplitude_mul
            self.centered_states[time_index:, lower_index:upper_index] = \
                self.centered_states[time_index:, lower_index:upper_index] + amplitude_mul

        elif typ == 'rise':

            from librosa.core import resample

            time_window = upper_index - lower_index

            pre_processed_states = self.centered_states[time_index:, lower_index:upper_index].transpose(-1, -2).numpy()

            upsampled_hb = resample(pre_processed_states, orig_sr=time_window, target_sr=time_window * resample_mul)
            distortion = np.where(upsampled_hb < 0)
            upsampled_hb[distortion] *= 1 / amplitude_mul
            upsampled_hb[not distortion] *= amplitude_mul

            window_diff = int(time_window * resample_mul) - time_window

            new_lower_index = lower_index - int(window_diff / 2)
            new_upper_index = upper_index + window_diff - int(window_diff / 2)

            noise = self.centered_observations[time_index:, new_lower_index:new_upper_index] - \
                    self.centered_states[time_index:, new_lower_index:new_upper_index]

            reprocessed = torch.from_numpy(upsampled_hb).transpose(-1, -2)

            self.centered_states[time_index:, new_lower_index:new_upper_index] = reprocessed
            self.centered_observations[time_index:, new_lower_index:new_upper_index] = reprocessed + noise


# class BaseECGSubset(Dataset):
#
#     def __init__(self, base_set: BaseECGLoader, index: int):
#
#         super(BaseECGSubset, self).__init__()
#
#         self.length = len(base_set) - index
#
#         datapoints = base_set.datapoints
#
#         self.dataset = base_set.dataset[:index]
#


