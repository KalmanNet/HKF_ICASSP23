import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from Filters.KalmanSmoother import KalmanFilter, KalmanSmoother
from PriorModels.BasePrior import BasePrior
from Dataloaders.BaseDataLoader import BaseECGLoader
from torch.utils.data.dataloader import DataLoader
from SystemModels.BaseSysmodel import BaseSystemModel
from tqdm import tqdm
from utils.Stich import stich, stich_with_interpolation
from utils.GetSubset import get_subset


class HKF_Pipeline:

    def __init__(self, prior_model: BasePrior, pat: int):
        self.prior_model = prior_model

        self.em_vars = ('Q', 'R')

        self.em_iterations = 50

        self.smoothing_window_Q = -1
        self.smoothing_window_R = -1

        self.n_residuals = 5

        self.number_sample_plots = 10

        self.show_results = False

        self.create_plot = True

        self.pat = pat

        self.num_channels = -1

        self.T = -1

        if not f'Patient_{self.pat}' in os.listdir('Plots'):
            os.mkdir(f'Plots/Patient_{self.pat}')

        self.folder = f'Patient_{self.pat}/'

    def fit_prior(self, prior_model: BasePrior, prior_set: BaseECGLoader) -> (BaseSystemModel, int):
        print('--- Fitting prior ---')

        prior_set_length = len(prior_set)

        observations, _ = next(iter(DataLoader(prior_set, batch_size=prior_set_length)))

        prior_model.fit(observations)

        sys_model = prior_model.get_sys_model()

        self.plot_prior(sys_model)

        return sys_model, prior_set_length

    def em_prior(self, intra_HKF: KalmanSmoother, prior_set: BaseECGLoader, T: int) -> KalmanSmoother:
        print('--- Estimating QR ---')

        torch.manual_seed(42)

        initial_q_2 = torch.rand(1).item()
        initial_r_2 = torch.rand(1).item()

        for n, (observation, _) in enumerate(prior_set):
            intra_HKF.init_mean(observation)

            intra_HKF.init_mean(observation[0].reshape(-1, 1))

            q = initial_q_2 if n == 0 else None
            r = initial_r_2 if n == 0 else None

            # Perform EM
            intra_HKF.em(observations=observation, states=None, num_its=self.em_iterations,
                         q_2_init=q, r_2_init=r,
                         T=T,
                         smoothing_window_Q=self.smoothing_window_Q, smoothing_window_R=self.smoothing_window_R
                         )

        return intra_HKF

    def plot_prior(self, prior_sys_model: BaseSystemModel):
        random_channel = np.random.randint(0, prior_sys_model.m)
        t = np.linspace(0, 1, prior_sys_model.T)
        prior_sys_model.init_sequence(torch.zeros(prior_sys_model.m, 1), torch.eye(prior_sys_model.m))
        prior_sys_model.generate_sequence(prior_sys_model.T)
        plt.plot(t, prior_sys_model.x[:, random_channel, 0])
        plt.xlabel('Timesteps')
        plt.ylabel('Amplitude [mV]')
        plt.grid()
        plt.title(f'Estimated prior of channel {random_channel}')
        plt.savefig(f'Plots/prior_plot.pdf')
        plt.show()

    def init_parameters(self, em_vars: tuple = ('R', 'Q'),
                        em_iterations: int = 50,
                        smoothing_window_Q: int = -1,
                        smoothing_window_R: int = -1,
                        n_residuals: int = 5,
                        number_sample_plots: int = 10,
                        create_plot: bool = True,
                        show_results: bool = False
                        ) \
            -> None:
        """
        Initialize parameters for both the inner and the outer KF/KS
        :param em_vars: List of variables to perform EM on
        :param smoothing_window_Q: Size of the window that is used to average Q in the EM-step
        :param smoothing_window_R: Size of the window that is used to average R in the EM-step
        :param n_residuals: Number of residuals used to update \mathcal{Q} in the ML-estimate step
        :return: None
        """
        self.em_vars = em_vars
        self.em_iterations = em_iterations

        self.smoothing_window_Q = smoothing_window_Q
        self.smoothing_window_R = smoothing_window_R

        self.n_residuals = n_residuals

        self.number_sample_plots = number_sample_plots

        self.create_plot = create_plot

        self.show_results = show_results and create_plot

    def init_filters(self, sys_model: BaseSystemModel, em_vars: tuple, test_set_length: int) \
            -> (KalmanSmoother, KalmanFilter):
        """
        Initialize the intra- and inter-filters
        """
        raise NotImplementedError

    def intra_operation(self, intra_hkf: KalmanSmoother, observation: torch.Tensor, T: int, m: int) \
            -> (torch.Tensor, torch.Tensor):
        """
        Perform the intra-operation
        """
        raise NotImplementedError

    def inter_operation(self, inter_hkf: KalmanFilter or list, observation: torch.Tensor, covariances: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        """
        Perform the inter-operation
        """
        raise NotImplementedError

    def run(self, prior_set: BaseECGLoader, test_set: BaseECGLoader) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        prior_set_length = len(prior_set)
        test_set_length = len(test_set)

        self.num_channels = prior_set.dataset.num_channels
        self.T = prior_set.dataset.datapoints

        # Set up internal system model
        intra_sys_model, _ = self.fit_prior(self.prior_model, prior_set)

        # Get parameters
        T = intra_sys_model.T
        m = intra_sys_model.m
        n = intra_sys_model.n
        num_channels = m

        # Set up data arrays
        full_intra_means = torch.empty(test_set_length, T, m)
        full_inter_means = torch.empty(test_set_length, T, m)

        # Set up loss arrays
        losses_intra = torch.empty(test_set_length)
        losses_inter = torch.empty(test_set_length, num_channels)
        loss_fn = torch.nn.MSELoss(reduction='mean')

        # Initialize the internal smoother and the external filters
        intra_HKF, inter_HKF = self.init_filters(intra_sys_model, self.em_vars, test_set_length)

        # Perform Prior QR learning
        intra_HKF = self.em_prior(intra_HKF, prior_set, T)

        # Set up iteration counter
        iterator = tqdm(test_set, desc='Hierarchical Kalman Filtering')

        for n, (observation, state) in enumerate(iterator):
            # Perform intra-operation
            intra_means, intra_covariances = self.intra_operation(intra_HKF, observation, T, m)
            # Perform inter-operation
            inter_means, inter_covariances = self.inter_operation(inter_HKF, intra_means, intra_covariances)

            # Save to result array
            full_intra_means[n] = intra_means
            full_inter_means[n] = inter_means

            # Calculate losses
            losses_intra[n] = loss_fn(intra_means, state)
            losses_inter[n] = loss_fn(inter_means, state)

        # Print losses
        mean_intra_loss = losses_intra.mean()
        mean_inter_loss = losses_inter.mean()

        mean_intra_loss_db = 10 * torch.log10(mean_intra_loss)
        mean_inter_loss_db = 10 * torch.log10(mean_inter_loss)

        print(f'Mean loss intra kalman smoother: {mean_intra_loss_db.item()}[dB]')
        print(f'Mean loss inter kalman filter: {mean_inter_loss_db.item()}[dB]')


        single_intra_plot_data = full_intra_means[-self.number_sample_plots:]
        single_inter_plot_data = full_inter_means[-self.number_sample_plots:]

        single_plot_data = [single_intra_plot_data, single_inter_plot_data]
        labels = ['Intra-HB Smoother', 'HKF']

        # Plot results
        single_observations_plot_data, single_state_plot_data = test_set[len(test_set) - self.number_sample_plots:len(test_set)]

        # Flag to see if the entire patient data is used, done to speed up debugging
        full_set_flag = test_set.dataset.centered_states.shape[0] == test_set_length + prior_set_length

        # Start and end index of the centered test-set, done to speed up debugging
        end_index_test_est = prior_set_length + test_set_length if not full_set_flag else -1

        # The start index of the un-centered test-set
        start_index_label = test_set.dataset.start_indices[prior_set_length]
        end_index_label = test_set.dataset.start_indices[end_index_test_est]

        # Get the overlaps for stiching
        overlaps = test_set.dataset.overlaps[0][prior_set_length:test_set_length + prior_set_length]
        # Overlap of the last HB
        extra_overlap = test_set.dataset.overlaps[0][
            end_index_test_est] if not full_set_flag else test_set.dataset.datapoints

        # True observations and states
        test_set_obs = test_set.dataset.observations[:, start_index_label:end_index_label + extra_overlap]
        test_set_states = test_set.dataset.dataset[:, start_index_label:end_index_label + extra_overlap]

        # Stiched filter results
        stiched_intra_means = stich_with_interpolation(full_intra_means, overlaps)
        stiched_inter_means = stich_with_interpolation(full_inter_means, overlaps)

        consecutive_obs_plot_data = test_set_obs[0, int(-self.number_sample_plots * test_set.dataset.datapoints):]
        consecutive_state_plot_data = test_set_states[0, int(-self.number_sample_plots * test_set.dataset.datapoints):]

        consecutive_intra_plot = stiched_intra_means[int(-self.number_sample_plots * test_set.dataset.datapoints):]
        consecutive_inter_plot = stiched_inter_means[int(-self.number_sample_plots * test_set.dataset.datapoints):]

        consecutive_plot_data = [consecutive_intra_plot, consecutive_inter_plot]

        if self.create_plot:
            self.plot_results(single_observations_plot_data,
                              single_state_plot_data,
                              single_plot_data,
                              consecutive_obs_plot_data,
                              consecutive_state_plot_data,
                              consecutive_plot_data,
                              labels,
                              test_set.dataset.fs
                              )

        return full_intra_means, full_inter_means, losses_intra, losses_inter

    def plot_results(self,
                     single_observations_plot_data: torch.Tensor,
                     single_state_plot_data: torch.Tensor = None,
                     single_plot_data: list = None,
                     consecutive_obs_plot_data: torch.Tensor = None,
                     consecutive_state_plot_data: torch.Tensor = None,
                     consecutive_plot_data: list or None = None,
                     labels: list = 'results',
                     fs: float = 1
                     ) -> None:

        """
        Plot filtered samples as well as the observation and the state
        observations: The observed signal with shape (samples, Time, channels)
        states: The ground truth signal with shape (samples, Time, channels)
        """

        folder = self.folder

        samples, T, channels = single_observations_plot_data.shape

        # Time steps for x-axis
        t = np.arange(start=0, stop=1, step=1 / T)

        # Choose which channel to plot
        channel = 0

        # A list of distinguishable colors
        distinguishable_color = ['#00998F', '#0075DC', '#fff017', '#5EF1F2', '#000075', '#911eb4']


        # Define font sizes
        legend_font_size = 15
        tick_size = 16
        title_size = 16
        label_size = 16


        # Check if ground truth state are provided
        if single_state_plot_data is None:
            state_flag = False
            single_state_plot_data = [None for _ in range(samples)]
        else:
            state_flag = True

        # Plot the multi-figure plots and single figure plots
        for j, (observation, state) in enumerate(zip(single_observations_plot_data, single_state_plot_data)):

            # Create figure and axes for single signal plots
            fig_single, ax_single = plt.subplots(figsize=(16, 9), dpi=120)
            single_figure_no_windows, ax_single_no_window = plt.subplots(figsize=(16, 9), dpi=120)

            # Plot the state if it is available
            if state is not None:
                ax_single.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')
                ax_single_no_window.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')

            # Plot observations
            ax_single.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)
            ax_single_no_window.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)

            # Plot the given results
            for i, (result, label) in enumerate(zip(single_plot_data, labels)):
                color = distinguishable_color[i]

                ax_single.plot(t, result[j][..., channel].squeeze(), label=label, color=color)
                ax_single_no_window.plot(t, result[j][..., channel].squeeze(), label=label, color=color)


            # Add legends
            ax_single.legend(fontsize=1.5 * legend_font_size)
            ax_single_no_window.legend(fontsize=1.5 * legend_font_size)

            # Set labels
            ax_single.set_xlabel('Time Steps', fontsize=1.5 * label_size)
            ax_single.set_ylabel('Amplitude [mV]', fontsize=1.5 * label_size)

            ax_single_no_window.set_xlabel('Time Steps', fontsize=1.5 * label_size)
            ax_single_no_window.set_ylabel('Amplitude [mV]', fontsize=1.5 * label_size)

            # Set axis parameters
            ax_single.xaxis.set_tick_params(labelsize=1.5 * tick_size)
            ax_single.yaxis.set_tick_params(labelsize=1.5 * tick_size)

            ax_single_no_window.xaxis.set_tick_params(labelsize=1.5 * tick_size)
            ax_single_no_window.yaxis.set_tick_params(labelsize=1.5 * tick_size)

            # Set title
            ax_single_no_window.set_title('Filtered Signal Sample', fontsize=1.5 * title_size)

            # Start plotting the zoomed in axis
            ax_ins = ax_single.inset_axes([0.05, 0.5, 0.4, 0.4])

            # Plot states if available
            if state is not None:
                ax_ins.plot(t, state[..., channel], color='g')

            # Plot results
            for i, (result, label) in enumerate(zip(single_plot_data, labels)):
                color = distinguishable_color[i]
                ax_ins.plot(t, result[j][..., channel].squeeze(), label=label, color=color)

            # Make axis invisible
            ax_ins.get_xaxis().set_visible(False)
            ax_ins.get_yaxis().set_visible(False)

            # Set axis parameters
            x1, x2, y1, y2 = 0.4, 0.6, ax_single.dataLim.intervaly[0], ax_single.dataLim.intervaly[1]
            ax_ins.set_xlim(x1, x2)
            ax_ins.set_ylim(y1, y2)
            ax_ins.set_xticklabels([])
            ax_ins.set_yticklabels([])
            ax_ins.grid()

            # Make box around plot data
            ax_single.indicate_inset_zoom(ax_ins, edgecolor="black")

            # Save plots
            fig_single.savefig(f'Plots\\{folder}Single_sample_plot_{j}.pdf')
            single_figure_no_windows.savefig(f'Plots\\{folder}Single_sample_plot_no_window_{j}.pdf')

            # Show plot
            if self.show_results:
                fig_single.show()
                single_figure_no_windows.show()
            else:
                fig_single.clf()
                single_figure_no_windows.clf()
        del fig_single
        del single_figure_no_windows




        stacked_y_min = torch.min(consecutive_state_plot_data[..., channel])
        stacked_y_max = torch.max(consecutive_state_plot_data[..., channel ])

        smallest_result_y_axis = torch.inf
        largest_result_y_axis = -torch.inf

        for result in consecutive_plot_data:

            y_stacked_min_results = torch.min(result[..., channel])
            y_stacked_max_results = torch.max(result[..., channel])

            if y_stacked_min_results < smallest_result_y_axis:
                smallest_result_y_axis = y_stacked_min_results
            if y_stacked_max_results > largest_result_y_axis:
                largest_result_y_axis = y_stacked_max_results

        num_samples = self.number_sample_plots
        # Time steps for x-axis
        t_cons = np.linspace(start=0, stop= 10, num= consecutive_state_plot_data.shape[0])
        y_axis_min = min(stacked_y_min.item(), smallest_result_y_axis.item())
        y_axis_max = max(stacked_y_max.item(), largest_result_y_axis.item())

        # Get the proper number of signals to plot
        num_signal = 2 if state_flag else 1

        # Get figures and axes
        fig_con, ax_cons = plt.subplots(nrows=num_signal + len(consecutive_plot_data), ncols=1, figsize=(16, 9), dpi=120)
        plt.ion()

        # Set tight layout
        fig_con.set_tight_layout(True)

        # Plot observations
        ax_cons[0].plot(t_cons, consecutive_obs_plot_data[..., channel].squeeze(), label='Observations', color='r',
                        alpha=0.4)


        # Set labels
        ax_cons[0].set_xlabel('Time [s]', fontsize=label_size)
        ax_cons[0].set_ylabel('Amplitude [mV]', fontsize=label_size)

        # Titles
        title_cons = 'Observations'
        ax_cons[0].set_title(title_cons, fontsize=title_size)

        # Configure axis
        ax_cons[0].xaxis.set_tick_params(labelsize=tick_size)
        ax_cons[0].yaxis.set_tick_params(labelsize=tick_size)

        # Check if state are available
        if state_flag:
            # Plot states
            ax_cons[1].plot(t_cons, consecutive_state_plot_data[..., channel].squeeze(), label='Ground Truth', color='g')

            # Set labels
            ax_cons[1].set_xlabel('Time [s]', fontsize=label_size)
            ax_cons[1].set_ylabel('Amplitude [mV]', fontsize=label_size)

            # Title
            title_cons = 'Ground Truth'
            ax_cons[1].set_title(title_cons, fontsize=title_size)

            # Configure axis
            ax_cons[1].xaxis.set_tick_params(labelsize=tick_size)
            ax_cons[1].yaxis.set_tick_params(labelsize=tick_size)
            ax_cons[1].set_ylim([y_axis_min, y_axis_max])

        # Loop over all results
        for j, (result, label) in enumerate(zip(consecutive_plot_data, labels)):
            # Get the color
            color = distinguishable_color[j]

            # Plot data
            ax_cons[j + num_signal].plot(t_cons, result[..., channel].squeeze(), color=color)

            # Set labels
            ax_cons[j + num_signal].set_xlabel('Time [s]', fontsize=label_size)
            ax_cons[j + num_signal].set_ylabel('Amplitude [mV]', fontsize=label_size)

            # Title
            ax_cons[j + num_signal].set_title(label, fontsize=title_size)

            # Configure axis
            ax_cons[j + num_signal].xaxis.set_tick_params(labelsize=tick_size)
            ax_cons[j + num_signal].yaxis.set_tick_params(labelsize=tick_size)
            ax_cons[j + num_signal].set_ylim([y_axis_min, y_axis_max])

        # Save consecutive plot
        fig_con.savefig(f'Plots\\{folder}Consecutive_sample_plots.pdf')

        fig_con.show()
