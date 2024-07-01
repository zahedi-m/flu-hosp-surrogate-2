import yaml


def read_config_file(filename):
    with open(filename, "rb") as fp:
        params= yaml.load(fp, Loader=yaml.SafeLoader)

    return params
