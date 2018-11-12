import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

import examples.nova_examples.predator_prey as predator_prey
import examples.nova_examples.vanderpol_2d as vanderpol_2d
import examples.nova_examples.vanderpol_8d as vanderpol_8d
import examples.nova_examples.biology_2 as biology_2
import examples.nova_examples.lorentz_system as lorenzt_system
from examples.nova_examples import coupled_oscillators_10d
from examples.nova_examples import coupled_oscillators_15d
from examples.nova_examples import coupled_oscillators_20d
from examples.nova_examples import coupled_oscillators_25d
from examples.nova_examples import coupled_oscillators_30d

xml_path = "/home/dxli/workspace/model-checkers/nova/src/examples/spaceex_examples/vanderpol_2d.xml"
cfg_path = "/home/dxli/workspace/model-checkers/nova/src/examples/spaceex_examples/vanderpol_2d.cfg"
opfile_root = "../spaceex_examples"
namespace = '{http://www-verimag.imag.fr/xml-namespaces/sspaceex}'

cfg_dict = {
    'system': 'sys',
    'initially': None,
    'forbidden': '\"x0 <= 0\"',
    'scenario': 'stc',
    'directions': 'oct',
    'set-aggregation': 'chull',
    'sampling-time': '0.01',
    'time-horizon': '3',
    'iter-max': '100',
    'output-variables': '\"x0, x1\"',
    'output-format': 'INTV',
    'rel-err': '1.0E-12',
    'abs-err': '1.0E-13',
    'flowpipe-tolerance': '0.001'
}
cfg_dict = OrderedDict(sorted(cfg_dict.items(), key=lambda t: t[0]))


def get_flow_str(flow_list):
    new_flow = []
    num = 0

    for f in flow_list:
        f = f.replace("**", "^")
        new_flow.append('x{}\' == {}'.format(num, f))
        num += 1

    return ' &'.join(new_flow)


def make_param_main(idx):
    attrib = {'name': 'x{}'.format(idx),
              'type': 'real',
              'local': 'false',
              'd1': '1',
              'd2': '1',
              'dynamics': 'any'
              }
    new_elem = ET.Element('ns0:param', attrib=attrib)

    return new_elem


def make_param_sys(idx):
    attrib = {'name': 'x{}'.format(idx),
              'type': 'real',
              'local': 'false',
              'd1': '1',
              'd2': '1',
              'dynamics': 'any',
              'controlled': 'true'
              }
    new_elem = ET.Element('ns0:param', attrib=attrib)

    return new_elem


def make_key(idx):
    key = 'x{}'.format(idx)
    attrib = {'key': key}
    new_elem = ET.Element('ns0:map', attrib=attrib)

    new_elem.text = key

    return new_elem


def read_model(model):
    ha = model.define_ha()
    init = model.define_init_states(ha)[0][1].bounds

    mode = ha.modes['1']
    flow = mode.dynamics.dyn_str
    model_name = model.__name__.split('.')[-1]

    return flow, init, model_name


def make_init_str(init):
    lb = init[0]
    ub = init[1]

    rv = ''
    for idx in range(len(init[0])):
        val = '{} <= x{} <= {}'.format(lb[idx], idx, ub[idx])
        if len(rv) > 0:
            rv = '{} & {}'.format(rv, val)
        else:
            rv = val

    rv = '{} & {}'.format(rv, 'loc(main_1)==running')

    return "\"{}\"".format(rv)


def main():
    # model = vanderpol_2d
    # model = vanderpol_4d
    # model = vanderpol_6d
    # model = vanderpol_8d
    # model = brusselator
    # model = buckling_column
    # model = coupled_oscillators_5d
    # model = coupled_oscillators_10d
    # model = coupled_oscillators_15d
    # model = biology_1
    # model = biology_2
    # model = lorenzt_system
    # model = predator_prey
    # model = coupled_oscillators_15d
    # model = coupled_oscillators_20d
    # model = coupled_oscillators_25d
    model = coupled_oscillators_30d
    flow, init, model_name = read_model(model)

    dim = len(flow)

    # write .xml
    tree = ET.parse(xml_path)
    root = tree.getroot()

    flow_str = get_flow_str(flow)

    for component in root.findall(namespace + 'component'):
        component_id = component.attrib['id']

        # clear up params
        for param in component.findall(namespace + 'param'):
            component.remove(param)

        # add new params
        if component_id == 'main':
            for i in range(dim):
                component.insert(-1, make_param_main(i))
        elif component_id == 'sys':
            for i in range(dim):
                component.insert(-1, make_param_sys(i))

        # change key map
        for bind in component.findall(namespace + 'bind'):
            for mapkey in bind.findall(namespace + 'map'):
                bind.remove(mapkey)

            # insert new map
            for i in range(dim):
                bind.insert(-1, make_key(i))

        for location in component.findall(namespace + 'location'):
            flow = location.find(namespace + 'flow')
            flow.text = flow_str

    opxml_path = os.path.join(opfile_root, '{}.xml'.format(model_name))
    tree.write(opxml_path)

    # write .cfg
    opcfg_path = os.path.join(opfile_root, '{}.cfg'.format(model_name))
    with open(opcfg_path, 'w') as cfg_content:
        init_str = make_init_str(init)
        cfg_dict['initially'] = init_str

        for key, val in cfg_dict.items():
            cfg_content.write('{} = {}\n'.format(key, val))

if __name__ == '__main__':
    main()
