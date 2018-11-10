"""Python 2.7 script for demonstrating automatic parameter tuning using Flow* 2.1.0

make sure hybridpy is on your PYTHONPATH: hyst/src/hybridpy
make sure the path to the flow* binary is on the HYPYPATH environment variable: ~/tools/flowstar

Stanley Bak
Oct 2018
"""
import os
import sys

import hybridpy.hypy as hypy
import hypy_config


def run(input_model_file, unsafe_condition, order, step_size, timeout):
    """run flowstar with the given unsafe condition, order, and step size

    returns a 2-tuple: (is_safe, runtime_seconds)
    """

    # we will add an error condition, and then find parameters in Flow* that prove safety with the error condition
    output_model_file = 'out_flowstar.model'
    flowstar_hyst_param = '-orders {} -step {} -unsafe "{}" -cutoff 1e-9 -nooutput'.format(order, step_size,
                                                                                           unsafe_condition)

    e = hypy.Engine('flowstar', flowstar_hyst_param)

    e.set_input(input_model_file)

    print_stdout = True
    image_path = None

    #### enable these for debugging ####
    e.set_output('flowstar_models/' + input_model_file.split('/')[-1].split('.')[
        0] + '.model')  # output the generated Flow* model to this path
    # e.set_verbose(True) # print hyst verbose output
    # print_stdout=True # print hyst/tool output
    # image_path='out.png' # output image path (requires GIMP is setup according to Hyst README)

    print("{}".format(step_size))
    result = e.run(parse_output=True, image_path=image_path, print_stdout=print_stdout, timeout=timeout)

    # result is a dictionary object with the following keys:
    # 'code' - exit code - engine.SUCCESS if successful, an engine.ERROR_* code otherwise
    # 'hyst_time' - time in seconds for hyst to run, only returned if run_hyst == True
    # 'tool_time' - time in seconds for tool(+image) to run, only returned if run_tool == True
    # 'time' - total run time in seconds
    # 'stdout' - the list of lines anything produced to stdout, only returned if save_stdout == True
    # 'tool_stdout' - the list of lines the tool produced to stdout, only returned if save_stdout == True
    # 'hypy_stdout' - the list of lines hypy produces to stdout, only returned if save_stdout == True
    # 'output' - tool-specific processed output object, only returned if successful and parse_output == True

    if result['code'] != hypy.Engine.SUCCESS:
        return False, -1, result['code']
        # raise RuntimeError('Hypy Error: {}'.format(result['code']))

    runtime = result['tool_time']
    output = result['output']

    # The output object is an ordered dictionary, with:
    # 'terminated' -> True/False   <-- did errors occur during computation (was 'terminated' printed?)
    # 'mode_times' -> [(mode1, time1), ...]  <-- list of reach-tmes computed in each mode, in order
    # 'result' -> 'UNKNOWN'/'SAFE'/None  <-- if unsafe set is used and 'terminated' is false, this stores
    #                                      the text after "Result: " in stdout
    # 'safe' -> True iff 'result' == 'SAFE'
    # 'gnuplot_oct_data' -> the octogon data in the tool's gnuplot output
    # 'reachable_taylor_models' -> a list of TaylorModel objects parsed from out.flow

    is_safe = output['result'] == "SAFE"
    print("({}) ".format(output['result']))

    return is_safe, runtime, result['code']


def autotune(input_model_file, start_stepsize, incre_stepsize, unsafe_conditions, orders, timeout):
    # for a fixed order, find the maximum step size possible that still ensures safety
    res_path_root = 'autotune_results'
    filename = input_model_file.split('/')[-1].split('.')[0] + '.res'
    res_path = os.path.join(res_path_root, filename)

    with open(res_path, 'w') as opfile:
        for unsafe_condition in unsafe_conditions:
            min_runtime = 1e9
            optimal_setting = None
            min_num_incre_without_timeout = 0

            opfile.write("===== Start tuning for condition {} =====\n".format(unsafe_condition))
            for order in orders:
                num_incre = min_num_incre_without_timeout

                while True:
                    safe, runtime, result_code = run(input_model_file, unsafe_condition, order, start_stepsize + num_incre * incre_stepsize,
                                                     timeout)

                    if result_code == hypy.Engine.TIMEOUT_TOOL:
                        # timed out,
                        # increase the step-size.
                        opfile.write("Step size {} for TM {} timed out. Trying larger steps.\n".format(start_stepsize + num_incre * incre_stepsize, order))
                        num_incre += 1
                        min_num_incre_without_timeout = num_incre
                        continue
                    else:  # didn't time out. Finished within time limit.
                        if not safe:
                            # case 1: unsafe time step, try higher TM orders
                            opfile.write("Step size {} for TM {} is too large. Trying higher TM orders.\n".format(start_stepsize + num_incre * incre_stepsize, order))
                            break
                        else:
                            # case 2: safe time step
                            # update optimal setting
                            safe_runtime = runtime
                            if safe_runtime < min_runtime:
                                min_runtime = safe_runtime
                                max_stepsize = start_stepsize + num_incre * incre_stepsize
                                optimal_setting = max_stepsize, order

                            opfile.write("@ Great! Found a safe step size {} for TM orders={}, runtime={}. Trying larger steps.\n".format(start_stepsize + num_incre * incre_stepsize,
                                                                                                         order, safe_runtime))
                            # increase num steps
                            num_incre += 1

                if optimal_setting is None:
                    opfile.write('Didn\'t find safe step size for TM order {}.\n\n'.format(order))
                else:
                    opfile.write("Optimal setting for condition {} is TM order={}, "
                                 "stepsize={}, runtime={}.\n\n".format(unsafe_condition,
                                                                       optimal_setting[1],
                                                                       optimal_setting[0],
                                                                       min_runtime))


def main():
    """main entry point"""
    input_model_path = '../src/examples/spaceex_examples/'
    evaluation_configs = hypy_config.config_hypy()
    timeout = 4

    for key, val in evaluation_configs.items():
        input_model_file = os.path.join(input_model_path, val['input_model_file'])
        autotune(input_model_file, val['start_step_size'], val['incre_step_size'], val['unsafe_condition'], val['order'], timeout)


if __name__ == '__main__':
    main()
