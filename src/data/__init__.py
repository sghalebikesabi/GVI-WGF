from .simulated import SimulatedDataset
from .uci import UCIDataset
from .wrapper import NumpyLoader

DATASET_DICT = {
    "simulated": SimulatedDataset,
    "uci": UCIDataset,
}
