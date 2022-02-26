from lib import build_data
import numpy as np
from tqdm import tqdm


def stochastic_avg():
    num_trials = 100
    tot_loss = 0
    progress_bar = tqdm(total=num_trials)
    for i in range(num_trials):
        progress_bar.update()
        data, filament_distance, filament_diameter, loss, mu = build_data.dataset(
            length_scale=52.0, sigma_f=50.0, sigma_y=10.0)
        tot_loss += loss

    progress_bar.close()
    loss = (1 / num_trials) * tot_loss
    loss = loss.reshape((68, 68))

    return filament_distance, filament_diameter, loss


filament_distance, filament_diameter, loss = stochastic_avg()
np.save('filament_distance', filament_distance)
np.save('filament_diameter', filament_diameter)
np.save('loss_stochastic', loss)
