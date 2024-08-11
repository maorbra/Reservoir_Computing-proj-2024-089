import os


SCOPE_ADDRESS_PATTERN = "USB0::0x0957::0x179A::[A-Z0-9]*::INSTR"
WFG_ADDRESS_PATTERN = "USB0::0x0957::0x0407::[A-Z0-9]*::INSTR"

os.chdir(os.path.dirname(__file__))
DATA_DIR = os.path.join(".", "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
