"""Country dataset engines for cadis-dataset-engine."""

from .dk import DenmarkAdminEngine
from .gb import GreatBritainAdminEngine
from .it import ItalyAdminEngine
from .jp import JapanAdminEngine
from .kr import SouthKoreaAdminEngine
from .no import NorwayAdminEngine
from .se import SwedenAdminEngine
from .tw import TaiwanAdminEngine

__all__ = [
    "DenmarkAdminEngine",
    "GreatBritainAdminEngine",
    "ItalyAdminEngine",
    "JapanAdminEngine",
    "SouthKoreaAdminEngine",
    "NorwayAdminEngine",
    "SwedenAdminEngine",
    "TaiwanAdminEngine",
]
