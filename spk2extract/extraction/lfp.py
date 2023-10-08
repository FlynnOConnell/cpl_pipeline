import numpy as np
from pathlib import Path
from spk2extract.spk_io import spk_h5

path = Path().home() / 'spk2extract' / 'h5'
files = list(path.glob('*.h5'))
for file in files:
    h5_file = spk_h5.read_h5(path / file)
    lfp = np.array(h5_file["data"]["lfp"]["lfp"])

    x = 4