from pathlib import Path
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw


spike2filepath = Path().home() / 'data' / 'context'
files = list(spike2filepath.glob('*.smr'))[0]
spike2_file_path = files.as_posix()
s2e = se.CedRecordingExtractor(spike2_file_path, stream_id="0")
recording = s2e
sw.plot_timeseries(s2e)

plt.show()

x = 5