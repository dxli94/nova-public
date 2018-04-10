import os
import numpy as np


class Plotter:
    images = []

    def __init__(self, images, opvars):
        self.images = images
        self.opvars = opvars
        self.vertices_sorted = list(map(lambda im: self.sort_vertices(im), self.images))

    def sort_vertices(self, im):
        # corners = [(v[self.opvars[0]], v[self.opvars[1]]) for v in im.vertices]
        corners = im.vertices

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

    def save_polygons_to_file(self):
        if not os.path.exists('../out/'):
            os.makedirs('../out/')

        with open('../out/outfile.out', 'w') as opfile:
            for vertices in self.vertices_sorted:
                x, y = [elem[0] for elem in vertices], [elem[1] for elem in vertices]
                x.append(x[0])
                y.append(y[0])

                for xx, yy in zip(x, y):
                    opfile.write('%.5f %.5f' % (xx, yy) + '\n')
                opfile.write('\n')

    @staticmethod
    def plot_polygons(ipfile_path):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        print('Start reading file...')
        try:
            with open(ipfile_path) as ipfile:
                content = ipfile.read().strip('\n')
                polygons = content.split('\n\n')
                vertices_sorted = list(map(lambda poly: poly.split('\n'), polygons))
        except FileExistsError:
            print('File does not exist %s' % ipfile_path)
        print('Finished. \nStart plotting...')

        fig = plt.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x_{1}$')
        ax.set_ylabel('$x_{2}$')

        for vertices in vertices_sorted:
            x, y = [float(elem.split()[0]) for elem in vertices], [float(elem.split()[1]) for elem in vertices]
            mat = np.transpose(np.array([x, y]))
            poly1patch = patches.Polygon(mat, fill=False, edgecolor='red', )
            ax.add_patch(poly1patch)

        plt.autoscale(enable=True)
        print('Showing plot now.')
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the input file storing vertex.')
    args = parser.parse_args()
    file = args.path if args.path else '../out/outfile.out'

    Plotter.plot_polygons(file)
