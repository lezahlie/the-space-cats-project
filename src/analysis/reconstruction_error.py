"""
implement the reconstruction error analysis and kde plots
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolor
import matplotlib.lines as mline
import seaborn as sns

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.image import PeakSignalNoiseRatio

