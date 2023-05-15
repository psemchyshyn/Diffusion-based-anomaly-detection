import yaml

def get_reconstruct_config(config_path="anomaly_detection/reconstruction-config.yaml"):
    config = None
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config