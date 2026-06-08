def parse_model_config(path):
    """
    Parses YOLO config file and returns list of module definitions.
    Each block in cfg becomes one dictionary.
    """
    with open(path, "r") as file:
        lines = file.read().split("\n")

    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.strip() for x in lines]

    module_defs = []

    for line in lines:
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].strip()

            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0

        else:
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            module_defs[-1][key] = value

    return module_defs