import unittest
import subprocess
import json
from pathlib import Path

import numpy as np

from speech2text.friends_benchmark import RATE, audio_chunks, channel_filter, clean_caption, english_stream, fixed_chunks, normalize, parse_srt, retained_reference, retained_spans, validate_inputs


class FriendsBenchmarkTest(unittest.TestCase):
    def test_input_identity_does_not_depend_on_checkout_path(self):
        episode = {'episode': 'S01E01', 'audio_samples': 100, 'audio': '/new/path/audio.wav',
                   'sha256': {'audio.wav': 'audio', 'captions.json': 'captions'}}
        expected = {'S01E01': {'samples': 100, 'audio_sha256': 'audio', 'captions_sha256': 'captions'}}
        validate_inputs({'episodes': [episode]}, expected)
        for key, value in [('audio_sha256', 'different'), ('captions_sha256', 'different'), ('samples', 101)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs({'episodes': [episode]}, {'S01E01': {**expected['S01E01'], key: value}})

    def test_committed_best_boundaries_cover_the_excluded_scope(self):
        root = Path(__file__).parent
        plan = json.loads((root / 'friends_s01_boundaries.json').read_text())
        cuts = json.loads((root / 'friends_s01_exclusions.json').read_text())
        self.assertEqual(plan['inputs'], cuts['inputs'])
        self.assertEqual(len(plan['episodes']), 24)
        self.assertEqual(sum(map(len, plan['episodes'].values())), 585)
        for name, parts in plan['episodes'].items():
            spans = retained_spans(plan['inputs'][name]['samples'], cuts['episodes'][name])
            fixed_chunks(parts, spans)

    def test_ffmpeg_selects_named_fc_instead_of_other_channels(self):
        stream = {'channels': 6, 'channel_layout': '5.1(side)'}
        output = subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1',
            '-f', 'lavfi', '-i', 'aevalsrc=0.1|0.2|0.3|0.4|0.5|0.6:s=16000:d=0.01:c=5.1(side)',
            *channel_filter(stream, True), '-threads', '1', '-f', 'f32le', '-',
        ], check=True, capture_output=True, timeout=10)
        samples = np.frombuffer(output.stdout, dtype='<f4')
        self.assertEqual(len(samples), 160)
        np.testing.assert_allclose(samples, 0.3, atol=1e-6)
        self.assertEqual(channel_filter(stream, False), [])
        with self.assertRaises(ValueError):
            channel_filter({'channels': 2, 'channel_layout': 'stereo'}, True)

    def test_fixed_chunks_reject_gaps_duplicates_and_excluded_audio(self):
        spans = [(10 * RATE, 70 * RATE), (90 * RATE, 100 * RATE)]
        parts = [{'start': 10, 'end': 70, 'error': None}, {'start': 90, 'end': 100, 'error': None}]
        self.assertEqual(fixed_chunks(parts, spans), spans)
        for bad in [parts[:1], parts + [parts[-1]], [dict(parts[0], start=9), parts[1]],
                    [dict(parts[0], end=71), parts[1]], [dict(parts[0], error='failed'), parts[1]]]:
            with self.assertRaises(ValueError):
                fixed_chunks(bad, spans)

    def test_exclusions_preserve_cold_open_and_final_dialogue_without_stitching(self):
        cuts = [{'start': 20, 'end': 70}, {'start': 150, 'end': 160}]
        spans = retained_spans(160 * RATE, cuts)
        self.assertEqual(spans, [(0, 20 * RATE), (70 * RATE, 150 * RATE)])
        audio = np.ones(160 * RATE, dtype='float32')
        chunks = [(left + start, left + end) for left, right in spans for start, end in audio_chunks(audio[left:right])]
        self.assertEqual(sum(end - start for start, end in chunks), 100 * RATE)
        self.assertTrue(all(end <= 20 * RATE or 70 * RATE <= start < end <= 150 * RATE for start, end in chunks))
        cues = [{'start': 10, 'end': 12, 'text': 'Cold open'}, {'start': 30, 'end': 40, 'text': 'Song'}, {'start': 145, 'end': 149, 'text': 'Last joke'}]
        self.assertEqual(retained_reference(cues, cuts), 'Cold open Last joke')
        with self.assertRaises(ValueError):
            retained_reference([{'start': 19, 'end': 21, 'text': 'Crossing speech'}], cuts)

    def test_invalid_exclusions_fail_before_evaluation(self):
        for intervals in [[(10, 5)], [(-1, 5)], [(0, 161)], [(0, 160)], [(10, 30), (20, 40)], [(float('nan'), 30)]]:
            with self.assertRaises(ValueError):
                retained_spans(160 * RATE, [{'start': start, 'end': end} for start, end in intervals])
        self.assertEqual(retained_spans(160 * RATE, []), [(0, 160 * RATE)])

    def test_mislabeled_original_track_is_distinguished_from_russian_dubbing(self):
        streams = [
            {'index': 1, 'codec_type': 'audio', 'tags': {'language': 'rus', 'title': 'Многоголосый (РТР)'}},
            {'index': 2, 'codec_type': 'audio', 'tags': {'language': 'rus', 'title': 'Оригинал AC3 5.1 @ 384 kbps'}},
        ]
        self.assertEqual(english_stream(streams, 'audio')['index'], 2)
        with self.assertRaises(ValueError):
            english_stream(streams + [streams[1]], 'audio')

    def test_sdh_labels_and_descriptions_are_not_spoken_words(self):
        raw = '[BOTH YELL]\nCHANDLER: <i>I’m sorry.</i>\n- Yes, yes!'
        self.assertEqual(clean_caption(raw), 'I’m sorry. Yes, yes!')

    def test_real_parentheses_and_repeated_speech_are_preserved(self):
        self.assertEqual(clean_caption('I said (yes), yes!'), 'I said (yes), yes!')
        self.assertEqual(normalize('Hey, Mon. It’s GREAT!'), 'hey mon it\x27s great')

    def test_group_speaker_labels_allow_lowercase_connectors(self):
        self.assertEqual(clean_caption('ROSS and MONICA: Hey, Joey.'), 'Hey, Joey.')
        self.assertEqual(clean_caption('CHANDLER and JOEY: Hockey!'), 'Hockey!')
        self.assertEqual(clean_caption('I agree and say: yes.'), 'I agree and say: yes.')

    def test_srt_bom_multiline_and_overlapping_display_times(self):
        text = '\ufeff1\r\n00:00:01,100 --> 00:00:03,000\r\nRACHEL: Oh,\r\nmy God.\r\n\r\n2\r\n00:00:02,500 --> 00:00:04,000\r\n[BOTH LAUGH]\r\n'
        cues = parse_srt(text)
        self.assertEqual((cues[0]['start'], cues[0]['end']), (1.1, 3.0))
        self.assertEqual(cues[0]['text'], 'Oh, my God.')
        self.assertEqual(cues[1]['text'], '')

    def test_bad_timestamps_are_rejected(self):
        for text in ['1\n00:00:02,000 --> 00:00:01,000\nHello', '1\n00:60:00,000 --> 00:61:00,000\nHello']:
            with self.assertRaises(ValueError):
                parse_srt(text)

    def test_chunking_covers_every_sample_once_and_finds_a_pause(self):
        audio = np.ones(125 * RATE, dtype='float32')
        audio[54 * RATE:55 * RATE] = 0
        boundaries = list(audio_chunks(audio))
        self.assertTrue(54 * RATE <= boundaries[0][1] <= 55 * RATE)
        self.assertEqual(boundaries[0][0], 0)
        self.assertEqual(boundaries[-1][1], len(audio))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(boundaries, boundaries[1:])))
        self.assertTrue(all(0 < end - start <= 60 * RATE for start, end in boundaries))
        np.testing.assert_array_equal(np.concatenate([audio[start:end] for start, end in boundaries]), audio)


if __name__ == '__main__':
    unittest.main()
