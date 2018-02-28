import os
import numpy as np


class Plotter:
    images = []

    def __init__(self, images, opvars):
        self.images = images
        self.opvars = opvars
        self.vertices_sorted = list(map(lambda im: self.sort_vertices(im), self.images))

    def sort_vertices(self, im):
        corners = [(v[self.opvars[0]], v[self.opvars[1]]) for v in im.vertices]

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

    def plot_polygons(self, flag_op=False):
        if flag_op:
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
        else:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            fig = plt.figure(1, dpi=90)
            ax = fig.add_subplot(111)
            for vertices in self.vertices_sorted:
                x, y = [elem[0] for elem in vertices], [elem[1] for elem in vertices]
                mat = np.transpose(np.array([x, y]))
                poly1patch = patches.Polygon(mat, fill=False, edgecolor='blue')
                ax.add_patch(poly1patch)

            plt.autoscale(enable=True)
            plt.show()
