import abc


class Data(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self, input):
        return

    @abc.abstractmethod
    def save(self, output, data):
        return