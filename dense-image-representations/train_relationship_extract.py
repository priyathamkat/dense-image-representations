from torchvision import transforms
from torch.utils.data import DataLoader

from data_processing.visual_genome import VisualGenomeDataset
import sys
sys.path.append('/cmlscratch/nehamk/segment-anything')
from segment_anything import sam_model_registry

sam = sam_model_registry['vit_b'](checkpoint='../sam_vit_b_01ec64.pth')
dataset = VisualGenomeRelationshipWithSAM(root_dir='/cmlscratch/nehamk/datasets/visual_genome', sam_model=sam)

