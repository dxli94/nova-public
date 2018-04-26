import json


tau = 0.01
time_horizon = 5
direction_type = 0
dim = 2
start_epsilon = 1e-9
dynamics = tuple(('0.1+0.01*(4-x1)+0.015*(2*9.81*x0)**0.5',
                  '0.015*(2*0.981*x0)**0.5-0.015*(2*9.81*x1)**0.5'))
vars = tuple(('x0', 'x1'))



d = dict()
d['sampling_time'] = tau
d['time_horizon'] = time_horizon
d['direction_type'] = direction_type
d['dim'] = dim
d['start_epsilon'] = start_epsilon
d['dynamics'] = dynamics
d['state_variables'] = vars
d['is_linear'] = [False, False]

print(json.dumps(d, indent=4, sort_keys=True))