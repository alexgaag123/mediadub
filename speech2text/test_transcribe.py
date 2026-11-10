import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from speech2text.transcribe import read_audio, save_transcript


class AudioIOTest(unittest.TestCase):
    def test_decode_and_save_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'silence.wav'
            with wave.open(str(source), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(bytes(3200))
            audio = read_audio(source)
            self.assertEqual(len(audio), 1600)
            np.testing.assert_array_equal(audio, 0)
            output = root / 'result'
            save_transcript({'text': 'Hello.'}, output)
            self.assertEqual((output / 'transcript.txt').read_text(), 'Hello.\n')
            with self.assertRaises(FileExistsError):
                save_transcript({'text': 'Replacement.'}, output)
            self.assertEqual((output / 'transcript.txt').read_text(), 'Hello.\n')


if __name__ == '__main__':
    unittest.main()
