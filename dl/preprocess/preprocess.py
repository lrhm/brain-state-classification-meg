import torch as t
import h5py
import os
import matplotlib.pyplot as plt
import ipdb
import pywt

def visualize_waves(data: t.Tensor):
    for i in range(data.shape[0]):
        plt.plot(data[i, :].numpy())
    plt.show()


def get_dataset_name(file_name: str):
    return "_".join(os.path.basename(file_name).split("_")[:-1])


def get_file_content(file_name: str):
    with h5py.File(file_name, "r") as f:
        raw_data = t.from_numpy(f[get_dataset_name(file_name)][:]).float()
    return raw_data


def extract_labels(f_name: str, data: t.Tensor):
    split_name = os.path.basename(f_name).split("_")
    task = "_".join(split_name[1:-2]) if ("task" in split_name) else split_name[0]
    subject = split_name[-2]
    return task, subject, data


def assign_labels(data: tuple[tuple[str, str, t.Tensor], ...]):
    # transposes data
    tasks, subjects, tensors = tuple(zip(*data))
    task_map = {key: i for i, key in enumerate(tuple(set(tasks)))}
    subject_map = {key: i for i, key in enumerate(tuple(set(subjects)))}
    # applies the maps to the data
    seq_size = tensors[0].shape[1]
    tasks = t.tensor(
        tuple(tuple(task_map[task] for _ in range(seq_size)) for task in tasks)
    ).unsqueeze(1)
    subjects = t.tensor(
        tuple(
            tuple(subject_map[subject] for _ in range(seq_size)) for subject in subjects
        )
    ).unsqueeze(1)
    merged_data = t.stack(tensors)
    merged_data_with_labels = t.cat((tasks, subjects, merged_data), dim=1)
    return merged_data_with_labels


def extract_preprocessed(
    i, j, labels: t.Tensor, data: t.Tensor, filter_freq: int, method="fourier"#None #"fourier"
):
    section = data[:, :, i:j]
    label_section = labels[:, :, i:j]
    plot = False
    if method == "fourier":
        #rand_start =  t.randint(2,1000, (1,))[0]
        rand_start = 0
        filter_freq = 1500
        f_section = t.fft.rfft(section, dim=2)[:, :, rand_start:rand_start+filter_freq]
        if plot:
            _, axis = plt.subplots(nrows=2, ncols=1)
            axis[0].set_title("Normalized Signal")
            axis[0].plot(section[3,3])
            axis[0].set(xlabel="Time", ylabel="Value [numeric]")
            axis[1].plot(t.abs(f_section[3,3][1:]))
            axis[1].set_title("Fourier")
            axis[1].set(xlabel="Frequency", ylabel="Magnitude")
            plt.show()

        section = t.view_as_real(section).view(section.shape[0], section.shape[1], -1)
        # ipdb.set_trace()    
        
    elif method == "wavelet":
        # computes wavelet
        section = t.from_numpy(pywt.cwt(section.numpy(), 4, "mexh", axis=2)[0]).squeeze(
            0
        )
        rand_start = t.randint(2,1000, (1,))[0]
        section = section[:, :, rand_start: rand_start + filter_freq]
        # section = t.cat((section[:,:,:20], section[:,:,-20:]), dim=2)
        # ipdb.set_trace()
        # visualize_waves(section[0, :10, :])
    section[:, :2] = label_section[:, :, : section.shape[-1]]
    return section


def window_data(
    data: t.Tensor, step_size: int, *, window_size: int = 1500, filter_freq: int = 69
):
    # creates a list of overlapping segments
    vars = t.var(data[:, 2:], dim=2).unsqueeze(-1)
    means = t.mean(data[:, 2:], dim=2).unsqueeze(-1)
    data[:, 2:] = (data[:, 2:] - means) / vars  # normalizes the data
    labels = data[:, :2]
    segments = []
    i = 0
    while i * step_size + window_size < data.shape[2]:
        segments.append(
            extract_preprocessed(
                i * step_size, i * step_size + window_size, labels, data, filter_freq,
            )
        )
        i += 1
    segments.append(extract_preprocessed(-window_size, None, labels, data, filter_freq))
    merged_segments = t.stack(segments, dim=2).permute(0, 2, 1, 3)
    normalized = normalize_freqs(
        merged_segments.reshape(
            -1, merged_segments.shape[2], merged_segments.shape[-1]
        )[:, :, 1:]
    )
    permuting_idxs = t.randperm(normalized.shape[0])
    normalized = normalized[permuting_idxs]
    """
    cutting_idx = int(0.2*normalized.shape[0])
    test = normalized[:cutting_idx]
    train = normalized[cutting_idx:]

    train_labels = train[:, :2, 0]
    train_data = train[:, 2:]
    test_labels = test[:, :2, 0]
    test_data = test[:, 2:]
    """
    # return {'train':(train_labels, train_data), 'test':(test_labels, test_data)}
    return normalized


def frequency_analisys(data: t.Tensor):
    means = t.mean(t.view_as_real(data), dim=0)
    vars = t.var(t.view_as_real(data), dim=0)
    # makes a plot with multiple subplots
    max_chs = 10
    _, axes = plt.subplots(nrows=max_chs, ncols=1)
    for i in range(max_chs):
        axes[i].plot(vars[i, :, 1])
    plt.show()


"""
def downsample(data: t.Tensor, cut_freq: int = 100) -> t.Tensor:
    downsampled_data = t.view_as_real(data)[:, :, :cut_freq].reshape(
        data.shape[0], data.shape[1], 2 * cut_freq
    )
    return data
"""


def normalize_freqs(windowed_data: t.Tensor, fancy=True, dumb=True):

    if dumb:
        windowed_data[:, 2:, :] = windowed_data[:, 2:, :] / 10**14
        return windowed_data


    if not fancy:
        maxs = t.max(windowed_data, dim=2)[0][:, 2:, None]
        mins = t.min(windowed_data, dim=2)[0][:, 2:, None]
    else:
        maxs = t.max(t.max(windowed_data, dim=0)[0], dim=0)[0][None, None]
        mins = t.min(t.min(windowed_data, dim=0)[0], dim=0)[0][None, None]
    windowed_data[:, 2:, :] = (windowed_data[:, 2:] - mins) / (maxs - mins)
    return windowed_data


def sub_preprocess(data_location="/mnt/dla3/Data_Ass3/Cross/train"):
    print("Preprocessing data...")
    files = tuple(os.path.join(data_location, f) for f in os.listdir(data_location))
    all_files = tuple(
        extract_labels(f, get_file_content(f)) for f in files if f.endswith(".h5")
    )
    labeled = assign_labels(all_files)
    result = window_data(labeled, step_size=500)
    print("Done!")
    return result


def preprocess(data_location="/mnt/dla3/Data_Ass3/Cross"):
    test_data_with_labels = t.cat(
        tuple(
            sub_preprocess(os.path.join(data_location, f))
            for f in os.listdir(data_location)
            if "test" in f
        )
    )
    test_labels = test_data_with_labels[:, :2, 0]
    test_data = test_data_with_labels[:, 2:]
    train_data_with_labels = t.cat(
        tuple(
            sub_preprocess(os.path.join(data_location, f))
            for f in os.listdir(data_location)
            if "train" in f
        )
    )
    train_labels = train_data_with_labels[:, :2, 0]
    train_data = train_data_with_labels[:, 2:]
    return {
        "train": (train_labels, train_data),
        "test": (test_labels, test_data),
    }


if __name__ == "__main__":
    preprocess()
