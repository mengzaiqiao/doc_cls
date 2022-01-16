import numbers
import os
import random
import time
import zipfile
from functools import wraps

import numpy as np
import pandas as pd

import GPUtil
import torch
from tabulate import tabulate

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def ensureDir(dir_path):
    """Ensure a dir exist, otherwise c

    reate

    Args:
        dir_path (str): the target dir
    Return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def print_transformer(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print("The BERT model has {:} different named parameters.\n".format(len(params)))

    print("==== Embedding Layer ====\n")

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== First Transformer ====\n")

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== Output Layer ====\n")

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


def get_device(config):
    """
        Get one gpu id that have the most available memory.
    Returns:
        (int, str): The gpu id (None if no available gpu) and the the device string (pytorch style).
    """
    if "device" in config:
        if config["device"] == "cpu":
            return (None, "cpu")
        elif "cuda:" in config["device"]:  # receive an string with "cuda:#"
            return (
                int(config["device"].replace("cuda:", "")),
                config["device"],
            )
        elif len(config["device"]) < 1:  # receive an int string
            return (
                int(config["device"]),
                "cuda:" + config["device"],
            )

    gpu_id_list = GPUtil.getAvailable(
        order="memory", limit=3
    )  # get the fist gpu with the lowest load
    if len(gpu_id_list) < 1:
        gpu_id = None
        device_str = "cpu"
    else:
        gpu_id = gpu_id_list[0]
        # need to set 0 if ray only specify 1 gpu
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split()) == 1:
                #  gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
                gpu_id = 0
                print("Find only one gpu with id: ", gpu_id)
                device_str = "cuda:" + str(gpu_id)
        # print(os.system("nvidia-smi"))
        else:
            print("Get a gpu with the most available memory :", gpu_id)
            device_str = "cuda:" + str(gpu_id)
    return gpu_id, device_str


def update_args(config, args):
    """Update config parameters by the received parameters from command line

    Args:
        config (dict): Initial dict of the parameters from JOSN config file.
        args (object): An argparse Argument object with attributes being the parameters to be updated.

    Returns:
        None
    """
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    print_dict_as_table(config, "Received parameters form command line (or default):")


def un_zip(file_name, target_dir=None):
    """Unzip zip files

    Args:
        file_name (str or Path): zip file path.
        target_dir (str or Path): target path to be save the unzipped files.

    Returns:
        None

    """
    if target_dir is None:
        target_dir = os.path.dirname(file_name)
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        print(f"unzip file {names} ...")
        zip_file.extract(names, target_dir)
    zip_file.close()


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table

    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.

    Returns:
        None

    """
    print("-" * 60)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 60)
    return tabulate(df, headers=columns, tablefmt="psql")


def initialize_folders(base_dir):
    """Initialize the whole directory structure of the project

    Args:
        base_dir (str): Root path of the project.

    Returns:
        None
    """

    configs = base_dir + "/configs/"
    datasets = base_dir + "/datasets/"
    checkpoints = base_dir + "/checkpoints/"
    results = base_dir + "/results/"
    logs = base_dir + "/logs/"
    processes = base_dir + "/processes/"
    runs = base_dir + "/runs/"

    for dir in [configs, datasets, checkpoints, results, processes, logs, runs]:
        if not os.path.exists(dir):
            os.makedirs(dir)


def get_random_rep(raw_num, dim):
    """
    Generate a random embedding from a normal (Gaussian) distribution.
    Args:
        raw_num: Number of raw to be generated.
        dim: The dimension of the embeddings.
    Returns:
        ndarray or scalar
        Drawn samples from the normal distribution.
    """
    return np.random.normal(size=(raw_num, dim))


def timeit(method):
    """Decorator for tracking the execution time for the specific method
    Args:
        method: The method need to timeit.

    To use:
        @timeit
        def method(self):
            pass
    Returns:
        None
    """

    @wraps(method)
    def wrapper(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(
                "Execute [{}] method costing {:2.2f} ms".format(
                    method.__name__, (te - ts) * 1000
                )
            )
        return result

    return wrapper

def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.
    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.
    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")


def save_to_csv(result, result_file):
    """
    Save a result dict to disk.

    Args:
        result: The result dict to be saved.
        result_file: The file path to be saved.

    Returns:
        None

    """
    print_dict_as_table(result)
    for k, v in result.items():
        result[k] = [v]
    result_df = pd.DataFrame(result)
    if os.path.exists(result_file):
        print(result_file, " already exists, appending result to it")
        total_result = pd.read_csv(result_file)
        total_result = total_result.append(result_df)
    else:
        print("Create new result_file:", result_file)
        total_result = result_df
    total_result.to_csv(result_file, index=False)


def set_seed(seed):
    """
    Initialize all the seed in the system

    Args:
        seed: A global random seed.

    Returns:
        None

    """
    if type(seed) != int:
        raise ValueError("Error: seed is invalid type")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def scale(values, target_min, target_max, source_min=None, source_max=None):
    """Scale the value of a numpy array "values"
    from source_min, source_max into a range [target_min, target_max]

    Parameters
    ----------
    values : Numpy array, required
        Values to be scaled.

    target_min : scalar, required
        Target minimum value.

    target_max : scalar, required
        Target maximum value.

    source_min : scalar, required, default: None
        Source minimum value if desired. If None, it will be the minimum of values.

    source_max : scalar, required, default: None
        Source minimum value if desired. If None, it will be the maximum of values.

    Returns
    -------
    res: Numpy array
        Output values mapped into range [target_min, target_max]
    """
    if source_min is None:
        source_min = np.min(values)
    if source_max is None:
        source_max = np.max(values)
    if source_min == source_max:  # improve this scenario
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values


def clip(values, lower_bound, upper_bound):
    """Perform clipping to enforce values to lie
    in a specific range [lower_bound, upper_bound]

    Parameters
    ----------
    values : Numpy array, required
        Values to be clipped.

    lower_bound : scalar, required
        Lower bound of the output.

    upper_bound : scalar, required
        Upper bound of the output.

    Returns
    -------
    res: Numpy array
        Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def intersects(x, y, assume_unique=False):
    """Return the intersection of given two arrays"""
    mask = np.in1d(x, y, assume_unique=assume_unique)
    x_intersects_y = x[mask]

    return x_intersects_y


def excepts(x, y, assume_unique=False):
    """Removing elements in array y from array x"""
    mask = np.in1d(x, y, assume_unique=assume_unique, invert=True)
    x_excepts_y = x[mask]

    return x_excepts_y


def safe_indexing(X, indices):
    """Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Borrowed from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis
    """
    if hasattr(X, "shape"):
        if hasattr(X, "take") and (
            hasattr(indices, "dtype") and indices.dtype.kind == "i"
        ):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def validate_format(input_format, valid_formats):
    """Check the input format is in list of valid formats
    :raise ValueError if not supported
    """
    if not input_format in valid_formats:
        raise ValueError(
            "{} data format is not in valid formats ({})".format(
                input_format, valid_formats
            )
        )

    return input_format


def estimate_batches(input_size, batch_size):
    """
    Estimate number of batches give `input_size` and `batch_size`
    """
    return int(np.ceil(input_size / batch_size))


# def save_to_csv(result_df, result_file):
#     """
#     Save a result dict to disk.
#     Args:
#         result: The result dict to be saved.
#         result_file: The file path to be saved.
#     Returns:
#         None
#     """
#     if os.path.exists(result_file):
#         print(result_file, " already exists, appending result to it")
#         total_result = pd.read_csv(result_file)
#         total_result = total_result.append(result_df)
#     else:
#         print("Create new result_file:", result_file)
#         total_result = result_df
#     print("saving result ...")
#     print(result_df)
#     total_result.to_csv(result_file, index=False)


def get_rng(seed):
    """Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "{} can not be used to create a numpy.random.RandomState".format(seed)
    )
