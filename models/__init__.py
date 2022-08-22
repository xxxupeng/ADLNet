from models.PSMNet.stackhourglass import PSMNet
from models.GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from models.PENet.ENet import ENet
from models.HSMNet.hsm import HSMNet

__models__ = {
    "PSMNet": PSMNet,
    "GwcNet_G": GwcNet_G,
    "GwcNet_GC": GwcNet_GC,
    "ENet":ENet,
    "HSMNet":HSMNet, 
}
