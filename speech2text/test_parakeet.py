import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from speech2text.parakeet import transcribe


class ParakeetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'input.mp4'
        self.source.touch()
        self.model = Mock(device=torch.device('cpu'), dtype=torch.float32)
        self.model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[4, 5]]))
        self.processor = Mock(return_value=BatchFeature({'input_features': torch.ones(1, 8, 128)}))

    @patch('speech2text.parakeet.read_audio')
    def test_transcribes_the_complete_original_audio_and_preserves_repeats(self, decoder):
        audio = np.ones(40 * 16000, dtype='float32')
        decoder.return_value = audio
        self.processor.batch_decode.return_value = ['  Hey Mon. Yes, yes!  ']
        result = transcribe(self.source, (self.model, self.processor))
        self.assertIs(self.processor.call_args.args[0], audio)
        self.assertEqual(self.processor.call_args.kwargs, {'sampling_rate': 16000, 'return_tensors': 'pt'})
        self.assertEqual(result['text'], 'Hey Mon. Yes, yes!')
        self.assertEqual(result['duration'], 40.0)
        self.assertEqual(set(result), {'input', 'model', 'model_revision', 'duration', 'text'})

    @patch('speech2text.parakeet.read_audio', return_value=np.zeros(48000, dtype='float32'))
    def test_empty_recognition_stays_empty(self, decoder):
        self.processor.batch_decode.return_value = ['']
        result = transcribe(self.source, (self.model, self.processor))
        self.assertEqual(result['text'], '')

    @patch('speech2text.parakeet.read_audio', return_value=np.ones(16000, dtype='float32'))
    def test_missing_model_result_is_an_error(self, decoder):
        self.processor.batch_decode.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'Expected one transcription'):
            transcribe(self.source, (self.model, self.processor))

    def test_missing_file_does_not_call_model(self):
        with self.assertRaises(FileNotFoundError):
            transcribe(self.source.parent / 'missing.wav', (self.model, self.processor))
        self.processor.assert_not_called()
        self.model.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
