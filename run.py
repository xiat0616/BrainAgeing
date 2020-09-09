from comet_ml import Experiment
import git
from easydict import EasyDict
import matplotlib
matplotlib.use('agg')
import importlib
import os
import json
import argparse
import logging
# Experiment
class experiment(object):
    def __init__(self, comet_exp=None):
        self.log = None
        self.comet_exp = comet_exp
        self.repo = git.Repo(search_parent_directories=True)

    def init_logging(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        logging.basicConfig(filename=config.folder + "/logfile.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
        logging.getLogger().addHandler(logging.StreamHandler())

        self.log = logging.getLogger()
        self.log.debug(config.items())
        self.log.info("---- Setting up experiment at " + config.folder + "----")

    def get_conf(self, args):
        config_dict = importlib.import_module("configuration."+args.config).get()
        config_dict["git_hash"] = self.repo.head.object.hexsha
        self.comet_exp.log_parameters(config_dict)
        config = EasyDict(config_dict)
        self.save_config(config)
        return config

    def save_config(self, conf):
        if not os.path.exists(conf.folder):
            os.makedirs(conf.folder)
        with open(conf.folder+"/experiment_configuration.json", "w") as outfile:
            json.dump(dict(conf.items()), outfile, sort_keys=True)

    def get_executor(self, conf):
        # Initialize model
        module_name = conf.model.split('.')[0]
        model_name = conf.model.split('.')[1]
        model = getattr(importlib.import_module("models."+module_name), model_name)(conf)
        model.build()

        # Initialize executor
        module_name = conf.executor.split('.')[0]
        model_name = conf.executor.split('.')[1]
        executor = getattr(importlib.import_module("model_executor."+module_name), model_name)(conf, model, self.comet_exp)
        return executor

    def read_console_parameters(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--config", default="", help="The experiment configuration file", required=True)
        parser.add_argument("--test", help="Evaluate the model on the test data", type=bool)
        parser.add_argument("--test_dataset", help="Override default test dataset", choices=["ADNI"])
        parser.add_argument("--split", help="Data split to run")
        parser.add_argument("--augment", help="Augment training data", type=bool)
        return parser.parse_args()

    def run(self):
        args = self.read_console_parameters()
        conf = self.get_conf(args)
        self.init_logging(conf)
        executor = self.get_executor(conf)
        executor.train()

if __name__ == "__main__":
    # Add the following code anywhere in your machine learning file
    comet_exp = Experiment(api_key="ADJUST to your own Key",
                           project_name="ADJUST to your own project name", workspace="ADJUST to your own workspace")
    exper = experiment(comet_exp)
    exper.run()
