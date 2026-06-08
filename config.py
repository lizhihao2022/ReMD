import argparse

parser = argparse.ArgumentParser(description='ReMD')
parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
parser.add_argument("--config", type=str, default="./template_configs/ns2d/remd.yaml")
