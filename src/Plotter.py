import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils.PPLHelper as PPLHelper

class Plotter:
    images = []

    def __init__(self, images, opvars):
        self.images = images
        self.opvars = opvars
        self.vertices_sorted = list(map(lambda im: self.sort_vertices(im), self.images))

    def sort_vertices(self, im):
        # corners = [(v[self.opvars[0]], v[self.opvars[1]]) for v in im.vertices]
        corners = PPLHelper.get_2dVert_from_poly(im, 2)

        # corners = [v[:2] for v in im.vertices]
        n = len(corners)
        cx = float(sum(x for x, y in corners)) / n
        cy = float(sum(y for x, y in corners)) / n
        cornersWithAngles = []
        for x, y in corners:
            an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
            cornersWithAngles.append((x, y, an))
        cornersWithAngles.sort(key=lambda tup: tup[2])

        return list(map(lambda ca: (ca[0], ca[1]), cornersWithAngles))

    def save_polygons_to_file(self, filename='../out/outfile.out'):
        with open(filename, 'w') as opfile:
            for vertices in self.vertices_sorted:
                x, y = [elem[0] for elem in vertices], [elem[1] for elem in vertices]
                x.append(x[0])
                y.append(y[0])

                for xx, yy in zip(x, y):
                    opfile.write('%.15f %.15f' % (xx, yy) + '\n')
                opfile.write('\n')

    @staticmethod
    def plot_polygons(filelist, xlabel, ylabel):
        # print('Start reading file...')
        if not isinstance(filelist, list):
            filelist = [filelist]

        colors = ['red', 'blue']
        linewidths = [0.5, 0.5]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)], linestyles[:len(filelist)]):
            # print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    polygons = content.split('\n\n')
                    vertices_sorted = list(map(lambda poly: poly.split('\n'), polygons))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)
            # print('Finished. \nStart plotting...')

            i = 0
            # stepsize = max(len(vertices_sorted) // 50, 1)
            stepsize = 5

            for vertices in vertices_sorted:
                if i % stepsize == 0:
                    x, y = [float(elem.split()[0]) for elem in vertices], [float(elem.split()[1]) for elem in vertices]
                    mat = np.transpose(np.array([x, y]))
                    poly1patch = patches.Polygon(mat, fill=False, edgecolor=color, linewidth=lw, linestyle=ls)
                    ax.add_patch(poly1patch)
                i+=1

        plt.autoscale(enable=True)

        return plt

    @staticmethod
    def plot_points_from_file(filelist, opdims, xlabel, ylabel):
        if not isinstance(filelist, list):
            filelist = [filelist]
        # print('Start reading file...')
        colors = ['red', 'blue']
        linewidths = [0.5, 0.5]
        linestyles = ['solid', 'dashed']

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        for ipfile_path, color, lw, ls in zip(filelist, colors[:len(filelist)], linewidths[:len(filelist)], linestyles[:len(filelist)]):
            # print(ipfile_path)
            try:
                with open(ipfile_path) as ipfile:
                    content = ipfile.read().strip('\n')
                    points = content.split('\n')
                    points = list(map(lambda p: p.split('\n')[0], points))
                    x = []
                    y = []

                    for p in points:
                        xy = p.split()
                        x.append(float(xy[opdims[0]]))
                        y.append(float(xy[opdims[1]]))
            except FileExistsError:
                print('File does not exist %s' % ipfile_path)

        plt.plot(x, y)
        plt.autoscale(enable=True)

    @staticmethod
    def plot_points(x, y, xlabel, ylabel):
        # print('Start reading file...')
        color = 'darkblue'
        linewidths = 0.5
        linestyles = 'solid'

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{}$'.format(xlabel))
        ax.set_ylabel('$x_{}$'.format(ylabel))

        plt.plot(x, y, color=color, ls=linestyles, lw=linewidths, alpha=0.5)
        plt.autoscale(enable=True)

    @staticmethod
    def plot_pivots(ipfile_path, opdims, color):
        # fig = plt.figure(1, dpi=90)
        # ax = fig.add_subplot(111)
        # ax.set_xlabel('$x_{1}$')
        # ax.set_ylabel('$x_{2}$')

        try:
            with open(ipfile_path) as ipfile:
                content = ipfile.read().strip('\n')
                points = content.split('\n')
                x = []
                y = []

                for point in points:
                    xy = point.split()
                    x.append(float(xy[opdims[0]]))
                    y.append(float(xy[opdims[1]]))
        except FileNotFoundError:
            print('File does not exist %s' % ipfile_path)

        plt.plot(x, y, 'o', color=color, markersize=8, alpha=0.5)
        plt.autoscale(enable=True)


    @staticmethod
    def save_plt(opfile):
        # plt.savefig(opfile, format='eps', dpi=500)
        plt.savefig(opfile, format='png', dpi=500)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the input file storing vertex.')
    parser.add_argument('--opdims', type=int, nargs=2, help='2 dimensions to output.')
    parser.add_argument('--type', type=int, help='0 for points, 1 for polygons. If 2, polygon from first; points from '
                                                 'second.')
    # parser.
    # directions = SuppFuncUtils.generate_directions(direction_type, 2)
    args = parser.parse_args()

    ipfile = args.path
    opdims = args.opdims

    assert ipfile, 'No input file specified!'
    assert opdims, 'No output dimension specified!'

    data_type = args.type
    model_name = ipfile.split('/')[-1].split('.')[0]

    if data_type == 1:
        Plotter.plot_polygons(ipfile, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
    elif data_type == 2:
        # Plotter.plot_polygons(filelist)
        Plotter.plot_points_from_file(['../out/simu.out'], opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
    elif data_type == 3:
        Plotter.plot_polygons(ipfile, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
        Plotter.plot_points_from_file(['../out/simu.out'], opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))
        print('Showing plot now.')
    else:
        # plot poly
        poly_path = os.path.join('../out/sfvals', model_name, '{}-{}'.format(*opdims))
        Plotter.plot_polygons(poly_path, xlabel=str(opdims[0]), ylabel=str(opdims[1]))

        # plot simulation
        simu_path = os.path.join('../out/simu_traj', '{}.simu'.format(model_name))
        Plotter.plot_points_from_file(simu_path, opdims, xlabel=str(opdims[0]), ylabel=str(opdims[1]))

        # plot scaling points
        Plotter.plot_pivots('../out/pivots.out', opdims, 'green')
        Plotter.plot_pivots('../out/sca_cent.out', opdims, 'yellow')

        print('Showing plot now.')
    plt.show()
