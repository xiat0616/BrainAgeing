import logging
from abc import abstractmethod

# from gans.loaders import loader_factory

log = logging.getLogger("basenet")


class BaseNet(object):
    """
    Base model for segmentation neural networks
    """
    def __init__(self, conf):
        self.model = None
        self.conf = conf
        # self.loader = None
        # if hasattr(self.conf, "dataset_name") and len(self.conf.dataset_name) > 0:
        #     self.loader = loader_factory.init_loader(self.conf.dataset_name)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def load_models(self):
        pass
