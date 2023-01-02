from Dataloaders import MIT_BIH_DataLoader, ProprietaryDataLoader
from utils.GetSubset import get_subset
from Pipelines.HKF_Pipeline_single_channel import Single_channel_HKF_Pipeline
from Pipelines.HKF_Pipeline_multichannel import Multi_channel_HKF_Pipeline
from PriorModels.TaylorPrior import TaylorPrior


import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    dataset = 'MIT-BIH'
    dataset = 'Proprietary'


    patient = 0

    number_of_datasamples = 300 if dataset == 'MIT-BIH' else 250  # ~360 for MIT, ~250 for proprietary
    snr_db = 0
    noise_color = 0

    dataloader = MIT_BIH_DataLoader if dataset == 'MIT-BIH' else ProprietaryDataLoader

    dataloader = dataloader(number_of_datasamples, patient, snr_db, noise_color)

    distortion_timestep = len(dataloader) - 10

    # dataloader.distort(100, 150, distortion_timestep, 'jump', amplitude_mul = 0.2, resample_mul=1)

    prior_loader, test_loader = get_subset(dataloader, 15)

    prior_model = TaylorPrior(channels=dataloader.num_channels, window_size=5)

    pipeline = Multi_channel_HKF_Pipeline(prior_model, pat=patient)
    pipeline.init_parameters(em_iterations=10, smoothing_window_R=-1, smoothing_window_Q=-1, create_plot=True,
                             n_residuals=5, number_sample_plots=10, show_results=False)


    pipeline.run(prior_loader, test_loader)
