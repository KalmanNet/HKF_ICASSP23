import torch


def moving_average(data: torch.Tensor, window_size: int = 7, axis: int = 1) -> torch.Tensor:
    """
    Calculate the moving-average along a given axis and with a given window size
    :param data: Data tensor
    :param window_size: Window Size
    :param axis: Axis to take the average along
    :return: Tensor containing the moving average along the axis
    """
    # Get the length of the
    data_length = data.shape[axis]

    # Create a buffer for the result
    data_buffer = []

    # Loop through the data
    for t in range(data_length):

        # Upper and lower window limits
        lower_limit = max(t - int(window_size/2), 0)
        upper_limit = min(t + int(window_size/2), data_length)

        # Create array of indices
        indices = torch.arange(lower_limit, upper_limit)

        # Get the data along the specified axis
        slices = torch.index_select(data, axis, indices)

        # Append to the data buffer
        data_buffer.append(slices.mean(axis))

    return torch.stack(data_buffer, axis)
