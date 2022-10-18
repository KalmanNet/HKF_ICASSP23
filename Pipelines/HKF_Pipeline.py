import matplotlib.pyplot as plt
import numpy as np
import torch
from Filters.IntraHKF import IntraHKF
from Filters.InterHKF import InterHKF
from PriorModels.BasePrior import BasePrior
from Dataloaders.BaseDataLoader import BaseECGLoader
from torch.utils.data.dataloader import DataLoader
from SystemModels.BaseSysmodel import BaseSystemModel
from tqdm import tqdm
from utils.Stich import stich


class HKF_Pipeline:

    def __init__(self, prior_model: BasePrior):
        self.prior_model = prior_model

        self.em_vars = ('Q', 'R')

        self.em_iterations = 50

        self.smoothing_window_Q = -1
        self.smoothing_window_R = -1

        self.n_residuals = 5

        self.number_sample_plots = 10

        self.show_results = False

    def fit_prior(self, prior_model: BasePrior, prior_set: BaseECGLoader) -> BaseSystemModel:
        print('--- Fitting prior ---')

        prior_set_length = len(prior_set)

        observations, _ = next(iter(DataLoader(prior_set, batch_size=prior_set_length)))

        prior_model.fit(observations)

        sys_model = prior_model.get_sys_model()

        return sys_model

    def init_parameters(self, em_vars: tuple = ('R', 'Q'), em_iterations: int = 50,
                        smoothing_window_Q: int = -1, smoothing_window_R: int = -1,
                        n_residuals: int = 5,
                        number_sample_plots: int = 10,
                        show_results: bool = False)\
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

        self.show_results = show_results

    def run(self, prior_set: BaseECGLoader, test_set: BaseECGLoader) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        # Set up internal system model
        intra_sys_model = self.fit_prior(self.prior_model, prior_set)

        # Get parameters
        test_set_length = len(test_set)
        T = intra_sys_model.T
        m = intra_sys_model.m
        n = intra_sys_model.n
        num_channels = m

        # Set up data arrays
        intra_smoother_means = torch.empty(test_set_length, T, m)
        inter_filter_means = torch.empty(test_set_length, T, m)

        # Set up loss arrays
        losses_intra_smoother = torch.empty(test_set_length)
        losses_inter_filter = torch.empty(test_set_length, num_channels)
        loss_fn = torch.nn.MSELoss(reduction='mean')

        # Initialize the internal smoother and the external filters
        intra_HKF = IntraHKF(intra_sys_model, self.em_vars)
        inter_HKFs = [InterHKF(T, self.em_vars) for _ in range(num_channels)]

        # Initial guess for the starting covariances
        torch.manual_seed(42)
        initial_q_2 = torch.rand(1).item()
        initial_r_2 = torch.rand(1).item()

        # Set up iteration counter
        iterator = tqdm(test_set, desc='Hierarchical Kalman Filtering')

        for n, (observation, state) in enumerate(iterator):

            # Only for the first heartbeat
            if n == 0:
                # Set the prior x_{0|0} to be the first observation point
                intra_HKF.init_mean(observation[0].reshape(-1, 1))

                # Perform EM
                intra_HKF.em(observations=observation, states=state, num_its=self.em_iterations,
                             q_2_init=initial_q_2, r_2_init=initial_r_2,
                             T=T,
                             smoothing_window_Q=self.smoothing_window_Q, smoothing_window_R=self.smoothing_window_R
                             )

            # Smooth internally with learned parameters
            smoother_means, smoother_covariances = intra_HKF.smooth(observation, T)

            # Set up means and covariances
            smoother_means = smoother_means.reshape(T, m)
            smoother_covariances = smoother_covariances.reshape(T, m, m)

            # Save to result array
            intra_smoother_means[n] = smoother_means

            # Calculate smoother loss
            losses_intra_smoother[n] = loss_fn(intra_smoother_means[n], state)

            # External filter for each channel
            for channel, inter_HKF in enumerate(inter_HKFs):

                # Initialize the filter for the first pass
                if n == 0:
                    inter_HKF.init_online(test_set_length)

                # Get internal smoother output as new input for the outer filter
                channel_smoother_mean = smoother_means[..., channel].reshape(T, 1)
                # Get internal smoother covariance as estimate for the observation noise
                channel_smoother_covariance = smoother_covariances[..., channel, channel]

                # Update \mathcal{R}_\tau using smoother error covariance
                inter_HKF.update_R(torch.eye(T) * channel_smoother_covariance)

                # ML update \mathcal{Q}_\tau
                inter_HKF.ml_update_q(channel_smoother_mean)

                # Get the output of the external KF
                inter_filter_mean = inter_HKF.update_online(channel_smoother_mean)

                # Save to result array
                inter_filter_means[n, :, channel] = inter_filter_mean.squeeze()

                # Calculate filter loss for channel
                losses_inter_filter[n, channel] = loss_fn(inter_filter_means[n, :, channel], state[:, channel])

        # Print losses
        mean_intra_loss = losses_intra_smoother.mean()
        mean_inter_loss = losses_inter_filter.mean()

        mean_intra_loss_db = 10 * torch.log10(mean_intra_loss)
        mean_inter_loss_db = 10 * torch.log10(mean_inter_loss)

        print(f'Mean loss intra kalman smoother: {mean_intra_loss_db.item()}[dB]')
        print(f'Mean loss inter kalman filter: {mean_inter_loss_db.item()}[dB]')

        # Plot results
        observations_plot_data, state_plot_data = test_set[-self.number_sample_plots:]

        intra_smoother_plot_data = intra_smoother_means[-self.number_sample_plots:]
        inter_filter_plot_data = inter_filter_means[-self.number_sample_plots:]

        plot_data = [intra_smoother_plot_data, inter_filter_plot_data]
        labels = ['Intra smoother means', 'Inter filter means']

        overlaps = test_set.dataset.overlaps[0]

        self.plot_results(observations_plot_data,
                          state_plot_data,
                          plot_data,
                          labels,
                          overlaps
                          )

        return intra_smoother_means, inter_filter_means, losses_intra_smoother, losses_inter_filter.mean(-1)

    def plot_results(self,
                     observations: torch.Tensor,
                     states: torch.Tensor = None,
                     results: list = None,
                     labels: list = 'results',
                     overlaps: list = None) -> None:

        """
        Plot filtered samples as well as the observation and the state
        observations: The observed signal with shape (samples, Time, channels)
        states: The ground truth signal with shape (samples, Time, channels)
        """

        samples, T, channels = observations.shape

        # Time steps for x-axis
        t = np.arange(start=0, stop=1, step=1 / T)

        # Choose which channel to plot
        channel = 0

        # A list of distinguishable colors
        distinguishable_color = ['#00998F', '#0075DC', '#fff017', '#5EF1F2', '#000075', '#911eb4']

        # Number of rows/columns for the multi plots
        n_rows = 2
        n_cols = 2

        # Define font sizes
        legend_font_size = 15
        tick_size = 16
        title_size = 16
        label_size = 16

        # Create multi-figure plots
        number_of_plot_required = max(int(np.ceil(samples / (n_cols * n_rows))), 1)
        multi_figures = [plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 9), dpi=120) for _ in
                         range(number_of_plot_required)]

        # Set layout to be tight
        for fig, _ in multi_figures:
            fig.set_tight_layout(True)
            fig.suptitle('Filtered Signal Samples')

        # Check if ground truth state are provided
        if states is None:
            state_flag = False
            states = [None for _ in range(samples)]
        else:
            state_flag = True

        # Plot the multi-figure plots and single figure plots
        for j, (observation, state) in enumerate(zip(observations, states)):

            # Create figure and axes for single signal plots
            fig_single, ax_single = plt.subplots(figsize=(16, 9), dpi=120)
            single_figure_no_windows, ax_single_no_window = plt.subplots(figsize=(16, 9), dpi=120)

            # Get multi figure axes from array
            fig_multi, ax_multi = multi_figures[int(j / (n_rows * n_cols))]

            # Format axes
            current_axes = ax_multi[int(j % (n_rows * n_cols) / n_rows), j % n_cols]
            current_axes.tick_params(labelsize=8)
            current_axes.xaxis.set_tick_params(labelsize=tick_size)
            current_axes.yaxis.set_tick_params(labelsize=tick_size)

            # Plot the state if it is available
            if state is not None:
                ax_single.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')
                ax_single_no_window.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')
                current_axes.plot(t, state[..., channel].squeeze(), label='Ground Truth', color='g')

            # Plot observations
            ax_single.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)
            ax_single_no_window.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)
            current_axes.plot(t, observation[..., channel].squeeze(), label='Observation', color='r', alpha=0.4)

            # Plot the given results
            for i, (result, label) in enumerate(zip(results, labels)):
                color = distinguishable_color[i]

                ax_single.plot(t, result[j][..., channel].squeeze(), label=label, color=color)
                ax_single_no_window.plot(t, result[j].squeeze(), label=label, color=color)

                current_axes.plot(t, result[j][..., channel].squeeze(), label=label, color=color)

            # Add legends
            ax_single.legend(fontsize=1.5 * legend_font_size)
            ax_single_no_window.legend(fontsize=1.5 * legend_font_size)
            current_axes.legend(fontsize=1.5 * legend_font_size)

            # Set labels
            ax_single.set_xlabel('Time Steps', fontsize=1.5 * label_size)
            ax_single.set_ylabel('Amplitude [mV]', fontsize=1.5 * label_size)

            current_axes.set_xlabel('Time Steps', fontsize=1.5 * label_size)
            current_axes.set_ylabel('Amplitude [mV]', fontsize=1.5 * label_size)

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
                ax_ins.plot(t, state[...,channel], color='g')

            # Plot results
            for i, (result, label) in enumerate(zip(results, labels)):
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
            fig_single.savefig(f'Plots\\Single_sample_plot_{j}.pdf')
            single_figure_no_windows.savefig(f'Plots\\Single_sample_plot_no_window_{j}.pdf')

            # Show plot
            if self.show_results:
                fig_single.show()
            else:
                fig_single.clf()

        # Save multi figures
        for n, (multi_fig, _) in enumerate(multi_figures):
            multi_fig.savefig(f'Plots\\Multi_sample_plot_{n}.pdf')

            # Show figure
            if self.show_results:
                fig_multi.show()
            else:
                fig_multi.clf()

        # Plot multiple HBs
        consecutive_beats = min(10, samples)

        # Stich together the observations
        stacked_observations = stich(observations[-consecutive_beats:,...,channel], overlaps[-consecutive_beats:])

        # Stich together the states
        if state_flag:
            stacked_states = stich(states[-consecutive_beats:,...,channel], overlaps[-consecutive_beats:])
            stacked_y_min = torch.min(stacked_states)
            stacked_y_max = torch.max(stacked_states)
        else:
            stacked_y_min = torch.inf
            stacked_y_max = -torch.inf

        stacked_results = []

        smallest_result_y_axis = torch.inf
        largest_result_y_axis = -torch.inf

        for result in results:

            stacked_results.append(stich(result[-consecutive_beats:][..., channel], overlaps[-consecutive_beats:]))
            y_stacked_min_results = torch.min(result[..., channel])
            y_stacked_max_results = torch.max(result[..., channel])

            if y_stacked_min_results < smallest_result_y_axis:
                smallest_result_y_axis = y_stacked_min_results
            if y_stacked_max_results > largest_result_y_axis:
                largest_result_y_axis = y_stacked_max_results

        # Time steps for x-axis
        t_cons = np.arange(start=0, stop=consecutive_beats, step=consecutive_beats / len(stacked_observations))
        y_axis_min = min(stacked_y_min.item(), smallest_result_y_axis.item())
        y_axis_max = max(stacked_y_max.item(), largest_result_y_axis.item())

        # Get the proper number of signals to plot
        num_signal = 2 if state_flag else 1

        # Get figures and axes
        fig_con, ax_cons = plt.subplots(nrows=num_signal + len(stacked_results), ncols=1, figsize=(16, 9), dpi=120)

        # Set tight layout
        fig_con.set_tight_layout(True)

        # Plot observations
        ax_cons[0].plot(t_cons, stacked_observations.squeeze(), label='Observations', color='r', alpha=0.4)

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
            ax_cons[1].plot(t_cons, stacked_states.squeeze(), label='Ground Truth', color='g')

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
        for j, (result, label) in enumerate(zip(stacked_results, labels)):
            # Get the color
            color = distinguishable_color[j]

            # Plot data
            ax_cons[j + num_signal].plot(t_cons, result.squeeze(), color=color)

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
        fig_con.savefig(f'Plots\\Consecutive_sample_plots.pdf')

        if self.show_results:
            fig_con.show()
        else:
            fig_con.clf()
