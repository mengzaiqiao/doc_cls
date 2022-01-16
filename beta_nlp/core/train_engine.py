import json
import os
import random
import string
import sys
from datetime import datetime

from tqdm import tqdm

import GPUtil
import ray
import torch
from beta_nlp.utils import logger
from beta_nlp.utils.common import ensureDir, print_dict_as_table, set_seed, update_args
from beta_nlp.utils.constants import MAX_N_UPDATE
from ray import tune
from tabulate import tabulate


class TrainEngine(object):
    """Training engine for all the models."""

    def __init__(self, args):
        """Init TrainEngine Class."""
        self.data = None
        self.train_loader = None
        self.monitor = None
        self.engine = None
        self.args = args
        self.config = self.prepare_env()

    def prepare_env(self):
        """Prepare running environment.

        * Load parameters from json files.
        * Initialize system folders, model name and the paths to be saved.
        * Initialize resource monitor.
        * Initialize random seed.
        * Initialize logging.
        """
        # Load config file from json
        with open(self.args.config_file) as config_params:
            print(f"loading config file {self.args.config_file}")
            config = json.load(config_params)

        # Update configs based on the received args from the command line .
        update_args(config, self.args)

        # obtain abspath for the project
        config["system"]["root_dir"] = os.path.abspath(config["system"]["root_dir"])

        # construct unique model run id, which consist of model name, config id and a timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
        config["system"]["model_run_id"] = (
            config["model"]["model"]
            + "_"
            + config["model"]["config_id"]
            + "_"
            + timestamp_str
            + "_"
            + random_str
        )

        # Initialize random seeds
        set_seed(config["system"]["seed"] if "seed" in config["system"] else 2020)

        # Initialize working folders
        self.initialize_folders(config)

        config["system"]["process_dir"] = os.path.join(
            config["system"]["root_dir"], config["system"]["process_dir"]
        )

        # Initialize log file
        config["system"]["log_file"] = os.path.join(
            config["system"]["root_dir"],
            config["system"]["log_dir"],
            config["system"]["model_run_id"],
        )
        logger.init_std_logger(config["system"]["log_file"])

        print("Python version:", sys.version)
        print("pytorch version:", torch.__version__)

        #  File paths to be saved
        config["model"]["run_dir"] = os.path.join(
            config["system"]["root_dir"],
            config["system"]["run_dir"],
            config["system"]["model_run_id"],
        )
        config["system"]["run_dir"] = config["model"]["run_dir"]
        print(
            "The intermediate running statuses will be reported in folder:",
            config["system"]["run_dir"],
        )

        config["system"]["tune_dir"] = os.path.join(
            config["system"]["root_dir"], config["system"]["tune_dir"]
        )

        def get_user_temp_dir():
            tempdir = os.path.join(config["system"]["root_dir"], "tmp")
            print(f"ray temp dir {tempdir}")
            return tempdir

        ray.utils.get_user_temp_dir = get_user_temp_dir

        #  Model checkpoints paths to be saved
        config["system"]["model_save_dir"] = os.path.join(
            config["system"]["root_dir"],
            config["system"]["checkpoint_dir"],
            config["system"]["model_run_id"],
        )
        ensureDir(config["system"]["model_save_dir"])
        print("Model checkpoint will save in file:", config["system"]["model_save_dir"])

        config["system"]["result_file"] = os.path.join(
            config["system"]["root_dir"],
            config["system"]["result_dir"],
            config["system"]["result_file"],
        )
        print("Performance result will save in file:", config["system"]["result_file"])

        print_dict_as_table(config["system"], "System configs")
        return config

    def initialize_folders(self, config):
        """Initialize the whole directory structure of the project."""
        dirs = [
            "log_dir",
            "result_dir",
            "process_dir",
            "checkpoint_dir",
            "run_dir",
            "tune_dir",
            "dataset_dir",
        ]
        base_dir = config["system"]["root_dir"]
        for directory in dirs:
            path = os.path.join(base_dir, config["system"][directory])
            if not os.path.exists(path):
                os.makedirs(path)

    def load_dataset(self):
        """Load dataset."""
        self.data = None
        pass

    def check_early_stop(self, engine, model_dir, epoch):
        """Check if early stop criterion is triggered.

        Save model if previous epoch have already obtained better result.

        Args:
            epoch (int): epoch num

        Returns:
            bool: True: if early stop criterion is triggered,  False: else
        """
        if epoch > 0 and self.eval_engine.n_no_update == 0:
            # save model if previous epoch have already obtained better result
            engine.save_checkpoint(model_dir=model_dir)

        if self.eval_engine.n_no_update >= MAX_N_UPDATE:
            # stop training if early stop criterion is triggered
            print(
                "Early stop criterion triggered, no performance update for {:} times".format(
                    MAX_N_UPDATE
                )
            )
            return True
        return False

    def _train(self, engine, train_loader, save_dir):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            self.eval_engine.train_eval(
                self.data.valid[0], self.data.test[0], engine.model, epoch
            )

    def tune(self, runable):
        """Tune parameters using ray.tune."""
        config = vars(self.args)
        if "tune" in config:
            config["tune"] = False
        if "root_dir" in config and config["root_dir"]:
            config["root_dir"] = os.path.abspath(config["root_dir"])
        else:
            config["root_dir"] = os.path.abspath("..")
        config["config_file"] = os.path.abspath(config["config_file"])
        print(config)
        tunable = self.config["tunable"]
        for para in tunable:
            if para["type"] == "choice":
                config[para["name"]] = tune.grid_search(para["values"])
            if para["type"] == "range":
                values = []
                for val in range(para["bounds"][0], para["bounds"][1] + 1):
                    values.append(val)
                config[para["name"]] = tune.grid_search(values)

        analysis = tune.run(
            runable,
            config=config,
            local_dir=self.config["system"]["tune_dir"],
            # temp_dir=self.config["system"]["tune_dir"] + "/temp",
            resources_per_trial={"cpu": 3, "gpu": 1},
        )
        df = analysis.dataframe()
        df.to_csv(
            os.path.join(
                self.config["system"]["tune_dir"],
                f"{self.config['system']['model_run_id']}_tune_result.csv",
            )
        )
        print(tabulate(df, headers=df.columns, tablefmt="psql"))

    def test(self):
        """Evaluate the performance for the testing sets based on the final model."""
        self.eval_engine.test_eval(self.data.test, self.engine.model)
