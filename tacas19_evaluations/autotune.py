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
    flowstar_hyst_param = '-orders {} -step {} -unsafe "{}" -cutoff 1e-9 -nooutput'.format(order, step_size, unsafe_condition)

    e = hypy.Engine('flowstar', flowstar_hyst_param)

    e.set_input(input_model_file)

    print_stdout = True
    image_path = None

    #### enable these for debugging ####
    e.set_output('flowstar_models/' + input_model_file.split('/')[-1].split('.')[0] + '.model')  # output the generated Flow* model to this path
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


def autotune(input_model_file, min_step_size, unsafe_conditions, order, timeout):
    # for a fixed order, find the maximum step size possible that still ensures safety
    res_path_root = 'autotune_results'
    filename = input_model_file.split('/')[-1].split('.')[0] + '.res'
    res_path = os.path.join(res_path_root, filename)
    with open(res_path, 'w') as opfile:
        for unsafe_condition in unsafe_conditions:
            for od in order:
                safe_runtime = None
                num_steps = 1

                msg = "Finding maximum safe Flow* step size for condition '{}' with TM order {}".format(unsafe_condition, od)
                print msg
                opfile.write(msg)

                while True:
                    try:
                        safe, runtime, result_code = run(input_model_file, unsafe_condition, od, num_steps * min_step_size, timeout)
                    except RuntimeError as e:
                        opfile.write(str(e) + '\n')

                    if result_code == hypy.Engine.TIMEOUT_TOOL:
                        # if times out, increase the step size.
                        num_steps *= 2
                        continue

                    elif not safe:
                        # didn't time out, found unsafe step size. Then decrease the step size.
                        low = num_steps / 2
                        high = num_steps

                        break

                    # found a safe step size, try larger step size.
                    safe_runtime = runtime
                    num_steps *= 2

                if safe_runtime is None:
                    msg = "Condition '{}' with order {} was unsafe even at minimum step size: {}\n".format(
                        unsafe_condition, od, min_step_size)
                    print msg
                    opfile.write(msg)
                    continue

                # binary search between high and low to find the boundary of safety
                # low is always safe, high is always unsafe
                msg = "Found unsafe step size. Doing binary search between {} and {}.".format(
                    low * min_step_size, high * min_step_size)
                print msg
                opfile.write(msg)
                while (high - low) > 1:
                    mid = (high + low) / 2

                    try:
                        safe, runtime, result_code = run(input_model_file, unsafe_condition, od, mid * min_step_size, timeout)
                    except RuntimeError as e:
                        opfile.write(str(e) + '\n')
                        break

                    if safe:
                        low = mid
                        safe_runtime = runtime
                    elif result_code == hypy.Engine.TIMEOUT_TOOL:
                        low = mid
                    else:
                        high = mid

                print ""
                msg = "Completed Analaysis for condition '{}' with order {}:".format(unsafe_condition, od)
                print msg
                opfile.write(msg)

                msg = "Largest safe step size was {} (runtime {} sec). Step size of {} was unsafe.\n".format(
                    low * min_step_size, safe_runtime, high * min_step_size)
                print msg
                opfile.write(msg)

def main():
    """main entry point"""
    input_model_path = '../src/examples/spaceex_examples/'
    evaluation_configs = hypy_config.config_hypy()
    timeout = 3

    for key, val in evaluation_configs.items():
        input_model_file = os.path.join(input_model_path, val['input_model_file'])
        autotune(input_model_file, val['min_step_size'], val['unsafe_condition'], val['order'], timeout)


if __name__ == '__main__':
    main()
