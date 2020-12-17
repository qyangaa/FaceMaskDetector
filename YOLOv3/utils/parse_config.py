import os


def parse_model_config(path):
    # Parse config file into list of maps
    file = open(path, 'r')
    lines = file.read().split('\n')  # split line by newline
    lines = [x for x in lines if x and not x.startswith('#')]  # remove comments
    lines = [x.rstrip().lstrip() for x in lines]  # remove fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # default batch_normalize for conv layers
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


if __name__ == '__main__':
    cwd = os.getcwd()
    parent_dir = os.path.join(cwd, os.pardir)
    config_dir = os.path.join(parent_dir, "config/yolov3.cfg")
    module_defs = parse_model_config(config_dir)
    print(module_defs)