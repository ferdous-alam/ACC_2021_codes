# import dependencies
from lib import implement_algorithm
from lib import plotPolicy
from lib import build_data
import numpy as np

length_scale = 52.0
sigma_f = 50.0
sigma_y = 10.0  # sigma_y maintains the stochasticity of the data

# Build dataset from physics based model
data, filament_distance, filament_diameter, loss, mu = build_data.dataset(
    length_scale, sigma_f, sigma_y)


