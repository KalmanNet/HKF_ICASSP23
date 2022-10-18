from Dataloaders import MIT_BIH_DataLoader, ProprietaryDataLoader
from utils.GetSubset import get_subset
from Pipelines.HKF_Pipeline import HKF_Pipeline
from PriorModels.TaylorPrior import TaylorPrior

if __name__ == '__main__':

    dataset = 'MIT-BIH'

    list_of_patients = [0]
    number_of_datasamples = 360 if dataset == 'MIT-BIH' else 250 # ~360 for MIT, ~250 for proprietary
    snr_db = 0
    noise_color = 0

    dataloader = MIT_BIH_DataLoader if dataset == 'MIT-BIH' else ProprietaryDataLoader

    dataloader = dataloader(number_of_datasamples, list_of_patients, snr_db, noise_color)

    prior_loader, test_loader = get_subset(dataloader, 10)

    prior_model = TaylorPrior(channels=dataloader.num_channels)

    pipeline = HKF_Pipeline(prior_model)
    pipeline.init_parameters(em_iterations=50, smoothing_window_R=-1, smoothing_window_Q=-1)

    pipeline.run(prior_loader, test_loader)
