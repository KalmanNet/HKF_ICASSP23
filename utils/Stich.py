import torch
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt

def stich(segmented_data: torch.Tensor, overlaps: list) -> torch.Tensor:
    stiched_results = []

    prev_overlap = 0

    weights = [1, 0]

    interpolation_dist = 10

    for (lower_segment, upper_segment, overlap) in zip(segmented_data[:-1], segmented_data[1:], overlaps[1:]):

        if overlap > 0:
            mean = (weights[0] * lower_segment[-overlap:] + weights[1] * upper_segment[:overlap]) / sum(weights)
            lower_segment[-overlap:] = mean

            stiched_results.append(lower_segment[prev_overlap:])

            prev_overlap = max(overlap, 0)

    last_segment = segmented_data[-1]
    last_overlap = overlaps[-1]

    stiched_results.append(last_segment[last_overlap:])

    stiched_results = torch.cat(stiched_results)

    return stiched_results


def stich_with_interpolation(segmented_data: torch.Tensor, overlaps: list) -> torch.Tensor:

    stiched_results = []

    prev_overlap = 0

    weights = [1, 0]

    interpolation_dist = 20

    for (lower_segment, upper_segment, overlap) in zip(segmented_data[:-1], segmented_data[1:], overlaps[1:]):
        if len(lower_segment.shape) == 1:
            channels = 1
        else:
            channels = lower_segment.shape[-1]

        if overlap > 0:
            mean = (weights[0] * lower_segment[-overlap:] + weights[1] * upper_segment[:overlap]) / sum(weights)
            lower_segment[-overlap:] = mean

            stiched_results.append(lower_segment[prev_overlap:].reshape(-1, channels))

            prev_overlap = overlap

        elif overlap < 0:

            if len(lower_segment.shape) == 1:
                lower_segment = lower_segment.unsqueeze(-1)
            channels = lower_segment.shape[-1]

            lower_interp = lower_segment[-interpolation_dist:]
            upper_interp = upper_segment[:interpolation_dist]

            x_arr = np.linspace(0, 1, 2*interpolation_dist - overlap)
            y_arr = np.empty((2*interpolation_dist, channels))
            y_arr[:interpolation_dist] = lower_interp.reshape(-1, channels)#.squeeze()
            y_arr[-interpolation_dist:] = upper_interp.reshape(-1, channels)#.squeeze()


            x_interp = np.concatenate((x_arr[:interpolation_dist], x_arr[-interpolation_dist:]))

            interpolation = []
            for channel in range(channels):

                spline = interp1d(x_interp, y_arr[:,channel], 'next')
                interp_points = torch.tensor(spline(x_arr))
                interpolation.append(interp_points)

            # if len(lower_segment.shape) > 1:
            interp_points = torch.stack(interpolation).T
            append_data = torch.cat((lower_segment, interp_points[interpolation_dist:interpolation_dist-overlap]))



            stiched_results.append(append_data[prev_overlap:].reshape(-1, channels))
            prev_overlap = 0

        else:
            stiched_results.append(lower_segment.reshape(-1, channels))

    last_segment = segmented_data[-1]
    last_overlap = max(overlaps[-1],0)

    stiched_results.append(last_segment[last_overlap:].reshape(-1, channels))

    stiched_results = torch.cat(stiched_results)

    return stiched_results