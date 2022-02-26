# import packages
import pandas as pd
from lib import runGP


def dataset(length_scale=52.0, sigma_f=50.0, sigma_y=10.0):
    """
    Use gaussian process regression (GPR) to build the dataset from the simulation data of
    phononic crystals. The fit of the data depends on three parameters for a GPR.
    length_scale:
    sigma_f:
    sigma_y:  This hyper-parameter defines how much noise will be added to the model. This
    helps to build data with stochastic noise where the mean and variance can be found from
    the gaussian process (stochastic process.)
    """
    # import data ---> Add column header
    col_names = ['filament_distance', 'filament_diameter', 'Loss']
    phononic_data = pd.read_csv('phononic_dataset_new.csv', names=col_names, header=None)
    simulation_data = pd.DataFrame(phononic_data)  # convert into data frame
    filament_distance = simulation_data.iloc[:, 0] + 400
    filament_diameter = simulation_data.iloc[:, 1]
    loss = simulation_data.iloc[:, 2]

    lossStochastic, mu = runGP.runGaussianProcess(length_scale, sigma_f, sigma_y)

    return simulation_data, filament_distance, filament_diameter, lossStochastic, mu
