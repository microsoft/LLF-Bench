from itertools import product

def set_nested_value(data_dict, keys, value):
    current_dict = data_dict
    for key in keys[:-1]:
        current_dict = current_dict.setdefault(key, {})
    current_dict[keys[-1]] = value

def generate_combinations_dict(input_dict):
    """ Turn a dict of lists into a list of dicts. """
    keys = list(input_dict.keys())
    value_combinations = product(*(input_dict[key] for key in keys))

    result = []
    for combination in value_combinations:
        new_dict = {keys[i]: combination[i] for i in range(len(keys))}
        result.append(new_dict)
    return result

def flatten_dict_of_lists(d, separator=':'):
    """ Flatten a nested dict where the values are lists into a dict of lists. """
    new_dict = {}
    for k,v in d.items():
        if type(v) is list:  # reaching the end of the tree
            new_dict[k] = v
        elif type(v) is dict:
            dd = flatten_dict_of_lists(v)  # a dict of lists
            for kk, vv in dd.items():
                new_dict[k+separator+kk] = vv
        else:
            raise ValueError(f'Each element must a list or a dict. Getting {type(v)}')
    return new_dict

def unflatten_dict_of_lists(d, separator=':'):
    """ This is the inverse of flatten_dict_of_lists. """
    new_dict = {}
    for k,v in d.items():
        keys = k.split(separator)
        current_dict = new_dict
        for key in keys[:-1]:
            current_dict = current_dict.setdefault(key, {})
        current_dict[keys[-1]] = v
    return new_dict


def batch_exp(fun, logger=None):
    """
        This decorator is used to generate a batch of experiments from a function.

        Example:

            Suppose we want to run the experiments with fun using aaa=[1,2,3]
            and b=[4,5,6], and a fixed core_fun, while printing(logging) the
            inputs and outputs of fun.

                def fun(a, b, core_fun):
                    aaa = a['a']['a']
                    return core_fun(aa, aaa, b)

            We can do the following:

                batch_fun = batch_exp(fun, logger=lambda inputs, outputs: print(inputs, outputs))
                batch_inputs = {
                    'a': {'a': [1,2,3]},
                    'b': [4,5,6],
                    'core_fun' : [core_fun]
                }
                batch_fun(**batch_inputs)
    """
    # TODO support return value
    def wrapper(**dict_of_lists_kwargs):
        flatten_kwargs = flatten_dict_of_lists(dict_of_lists_kwargs)
        combinations = generate_combinations_dict(flatten_kwargs)
        for inputs in combinations:
            inputs = unflatten_dict_of_lists(inputs)
            outputs = fun(**inputs)
            if logger is not None:
                logger(inputs, outputs)
    return wrapper


if __name__=='__main__':

    x = {'a':[1,2], 'b':{'c':[3,4], 'd':[5,6]}}
    print('original', x)
    print(flatten_dict_of_lists(x))
    print('recovered', unflatten_dict_of_lists(flatten_dict_of_lists(x)))