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
    parser.add_argument('--output', type=int, help='1, print images to outfile.out\n 0, print to file.')
    parser.add_argument('--opvars', type=int, nargs='*', help='two index of variables to plot. Indexing from 0! First'
                                                              'two dimensions (0, 1) by default.')

    args = parser.parse_args()
    if args.opvars is not None:
        args.opvars = tuple(args.opvars)
        if len(args.opvars) is not 2:
            parser.error("Incorrect number of output variables provided! Either omitted, or two are required!")
    else:
        args.opvars = (0, 1)
    return args


def main():
    args = parse_args()
    instance_file = args.path
    direction_type = args.dt
    time_horizon = args.horizon
    samp_freq = args.sf
    flag_op = args.output == 0

    try:
        assert all(elem is not None for elem in [instance_file, direction_type, time_horizon, samp_freq])
    except:
        raise RuntimeError("Invalid arguments.")

    data_reader = DataReader(path2instance=instance_file)

    sys_dynamics = data_reader.read_data()
    sys_dim = sys_dynamics.get_dim()
    directions = SuppFuncUtils.generate_directions(direction_type=direction_type,
                                                   dim=sys_dim)

    assert max(args.opvars) < sys_dim, "output variables' index out of system dimensionality."

    post_opt = PostOperator(sys_dynamics, directions, time_horizon, samp_freq)

    sf_mat = post_opt.compute_post()
    images = post_opt.get_images(opdims=args.opvars, sf_mat=sf_mat)

    plotter = Plotter(images, args.opvars)
    plotter.plot_polygons(flag_op)


if __name__ == '__main__':
    main()
