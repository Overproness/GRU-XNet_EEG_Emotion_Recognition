"""Synthetic checks that selective SET inspection skips outcomes/waveforms."""
import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_dens_set_metadata.py"
SPEC = importlib.util.spec_from_file_location("dens_header", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(data="sub-mit003_task-Emotion_eeg.fdt", compressed=True):
    channel = np.empty((1, 2), dtype=[("labels", object), ("X", object)])
    channel[0, 0] = ("Cz", 0.0)
    channel[0, 1] = ("Fz", 1.0)
    event = np.empty((1, 2), dtype=[("type", object), ("latency", object)])
    event[0, 0] = ("DO_NOT_DECODE", 12.0)
    event[0, 1] = ("DO_NOT_DECODE", 24.0)
    eeg = {"nbchan": 2, "pnts": 2500, "trials": 1, "srate": 250.0,
           "xmin": 0.0, "xmax": 9.996, "data": data,
           "ref": "Cz", "chanlocs": channel, "event": event,
           "untrusted_outcomes": "DO_NOT_DECODE", "etc": {"ratings": [1, 9]}}
    stream = io.BytesIO()
    savemat(stream, {"EEG": eeg}, do_compression=compressed)
    return stream.getvalue()


class SelectiveMetadataTest(unittest.TestCase):
    def test_compressed_and_plain_decode_only_allowlisted_values(self):
        for compressed in (True, False):
            result = MODULE.inspect_bytes(fixture(compressed=compressed))
            self.assertEqual(result["acquisition"]["srate"], 250)
            self.assertEqual(result["channel_labels"], ["Cz", "Fz"])
            self.assertEqual(result["schema"]["event"]["entry_count"], 2)
            self.assertEqual(result["schema"]["event"]["field_names"], ["type", "latency"])
            self.assertIn("untrusted_outcomes", result["skipped_fields"])
            self.assertNotIn("DO_NOT_DECODE", str(result))
            self.assertEqual(result["event_values_decoded"], 0)
            self.assertEqual(result["behavioral_outcome_values_decoded"], 0)

    def test_embedded_waveform_rejected(self):
        with self.assertRaisesRegex(ValueError, "Embedded EEG.data"):
            MODULE.inspect_bytes(fixture(data=np.zeros((2, 2500))))

    def test_wrong_waveform_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected external waveform"):
            MODULE.inspect_bytes(fixture(data="arbitrary_file.fdt"))

    def test_non_mat_and_truncated_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.inspect_bytes(b"not mat")
        with self.assertRaises(ValueError):
            MODULE.inspect_bytes(fixture()[:-40])


if __name__ == "__main__":
    unittest.main()
