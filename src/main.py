import argparse

import SuppFuncUtils

from DataReader import DataReader
from Plotter import Plotter
from PostOperator import PostOperator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to a (single) instance file.')
    parser.add_argument('--dt', type=int, help='direction type. 0 for box; 1 for octagonal.')
    parser.add_argument('--horizon', type=float, help='time horizon.')
    parser.add_argument('--sf', type=float, help='sampling frequency.')

    return parser.parse_args()


def main():
    args = parse_args()
    instance_file = args.path
    direction_type = args.dt
    time_horizon = args.horizon
    samp_freq = args.sf

    try:
        assert all(elem is not None for elem in [instance_file, direction_type, time_horizon, samp_freq])
    except:
        raise RuntimeError("Invalid arguments.")

    data_reader = DataReader(path2instance=instance_file)

    sys_dynamics = data_reader.read_data()
    directions = SuppFuncUtils.generate_directions(direction_type=direction_type,
                                                   dim=sys_dynamics.get_dyn_init_X0()[0].shape[1])

    post_opt = PostOperator(sys_dynamics, directions, time_horizon, samp_freq)

    sf_mat = post_opt.compute_post()
    images = post_opt.get_images(sf_mat=sf_mat)
    # plotter.plot(images)
    for image in images:
        print(image.vertices)

    plotter = Plotter(images)
    plotter.plot()


if __name__ == '__main__':
    main()
