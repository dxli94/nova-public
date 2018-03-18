import os

ip_dir = '../out'
op_dir = '../out_2'

for file in os.listdir(ip_dir):
    if file.startswith('README'):
        continue
    with open(os.path.join(ip_dir, file), 'r') as ipfile, open(os.path.join(op_dir, file), 'w') as opfile:
        for line in ipfile:
            if line.strip('\n').__len__() > 0:
                x, y = line.strip('\n').split()
                # x, y = xy
                opfile.write('%.5f %.5f' % (float(x), float(y)) + '\n')
