import os
import re


def process_env_vars_in_config(cfg):
    """
    Recursively process environment variables in config dictionaries.
    Replaces strings like ${ENV_VAR} with the value of ENV_VAR.
    """
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if isinstance(value, (dict, list)):
                cfg[key] = process_env_vars_in_config(value)
            elif isinstance(value, str):
                cfg[key] = replace_env_vars(value)
    elif isinstance(cfg, list):
        for i, item in enumerate(cfg):
            cfg[i] = process_env_vars_in_config(item)
    elif isinstance(cfg, str):
        cfg = replace_env_vars(cfg)
    return cfg


def replace_env_vars(string):
    """
    Replace ${ENV_VAR} in the string with the value of ENV_VAR.
    """
    pattern = r"\${([^}]*)}"
    matches = re.findall(pattern, string)

    for match in matches:
        env_var = os.environ.get(match)
        if env_var is not None:
            string = string.replace(f"${{{match}}}", env_var)

    return string
