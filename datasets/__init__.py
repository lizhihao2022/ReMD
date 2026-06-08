from .ns2d import NavierStokes2DDataset
from .ERA5 import ERA5Dataset
from .ERA5temperature import ERA5TemperatureDataset
from .ERA5wind import ERA5WindDataset
from .Ocean import OceanDataset

_dataset_dict = {
    "NavierStokes2D": NavierStokes2DDataset,
    "ERA5": ERA5Dataset,
    "ERA5temperature": ERA5TemperatureDataset,
    "ERA5wind": ERA5WindDataset,
    "Ocean": OceanDataset,
}
