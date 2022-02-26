import numpy as np
from lib import build_data


# Build stochastic model
length_scale = 52.0
sigma_f = 50.0
sigma_y = 10.0  # sigma_y maintains the stochasticity of the data
budget = 5
for i in range(budget):
    print('model built #{}'.format(2000+i))
    _, _, _, loss_stochastic, _ = build_data.dataset(length_scale, sigma_f, sigma_y)
    loss_stochastic = loss_stochastic.reshape(68, 68)
    np.save('stochastic_models/loss_stochastic{}'.format(2000+i), loss_stochastic)
