import torch


def stich(segmented_data: torch.Tensor, overlaps: list) -> torch.Tensor:
    stiched_results = []

    prev_overlap = 0

    weights = [1, 1]

    for (lower_segment, upper_segment, overlap) in zip(segmented_data[:-1], segmented_data[1:], overlaps[1:]):

        if overlap != 0:
            mean = (weights[0] * lower_segment[-overlap:] + weights[1] * upper_segment[:overlap]) / sum(weights)
            lower_segment[-overlap:] = mean

        stiched_results.append(lower_segment[prev_overlap:])

        prev_overlap = overlap

    last_segment = segmented_data[-1]
    last_overlap = overlaps[-1]

    stiched_results.append(last_segment[last_overlap:])

    stiched_results = torch.cat(stiched_results)

    return stiched_results
