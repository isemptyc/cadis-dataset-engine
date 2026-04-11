"""Country dataset engines for cadis-dataset-engine."""

from .be import BelgiumAdminEngine
from .dk import DenmarkAdminEngine
from .gb import GreatBritainAdminEngine
from .it import ItalyAdminEngine
from .jp import JapanAdminEngine
from .kr import SouthKoreaAdminEngine
from .nl import NetherlandsAdminEngine
from .no import NorwayAdminEngine
from .se import SwedenAdminEngine
from .tw import TaiwanAdminEngine

__all__ = [
    "BelgiumAdminEngine",
    "DenmarkAdminEngine",
    "GreatBritainAdminEngine",
    "ItalyAdminEngine",
    "JapanAdminEngine",
    "SouthKoreaAdminEngine",
    "NetherlandsAdminEngine",
    "NorwayAdminEngine",
    "SwedenAdminEngine",
    "TaiwanAdminEngine",
]
