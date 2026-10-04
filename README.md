![Automatic video translation and voice-over pipeline](project-banner.png)

Annual project for the Artificial Intelligence master's program, 2026/27
academic year.

[Русская версия](README_RU.md)

## Project Idea

The goal is to build an application that receives a video in the source
language and returns a translated version while preserving timings, separating
characters, and reproducing voices similar to the original ones.

First-semester user scenario:

1. A user sends a video to the Telegram bot.
2. The system extracts and processes the audio track.
3. It detects utterances and identifies the speakers.
4. It transcribes and translates the source speech.
5. It synthesizes the translation using the characters' voices.
6. It fits the generated speech into the original time intervals and rebuilds
   the video.
7. The bot returns the finished video and a short processing report.

## Team

| Group | Members | Area of responsibility | Output for the next group |
|---|---|---|---|
| Original content analysis | Alexey Neurov, Alexander Gaag | Audio preparation, enhancement, VAD, segmentation, diarization, ASR, and quality evaluation | Utterances with timings, voice IDs, character names, and source text |
| Translation and dubbing | Alexander Kovylev, Nikita Panov | Translation, voice cloning, speech synthesis, duration fitting, and final assembly | Translated and dubbed video |

Supervisor: Nikita Karagodin.

### Proposed Individual Responsibilities

The internal allocation may be refined as experiments progress. The initial
proposal is:

- **Alexey:** audio extraction and normalization, enhancement, VAD, integration
  of the first pipeline stage, data preparation, and reproducible execution;
- **Alexander G.:** diarization, ASR, manual ground-truth annotation,
  WER/CER/VAD/DER/SER evaluation, and error analysis;
- **Alexander K.:** context-aware translation that accounts for terminology,
  utterance length, and the original timing;
- **Nikita:** voice cloning/TTS, prosody, duration fitting, audio mixing, and
  final video assembly.

The whole team contributes to FastAPI and Telegram bot integration. Every stage
must have an owner, while the data contract between the two groups is agreed on
jointly.

## Target Architecture

```text
Video
  ↓
English audio track extraction
  ↓
5.1 → stereo/mono downmix
  ↓
Music and noise removal
  ↓
VAD: speech activity detection
  ↓
Utterance segmentation
  ↓
16 kHz diarization: speaker_1, speaker_2, ...
  ↓
16 kHz ASR: source text and word timings
  ↓
speaker_N → character mapping
  ↓
Timing-constrained translation
  ↓
Voice cloning and translated speech synthesis
  ↓
Tempo, loudness, and duration fitting
  ↓
Mixing with the original music and effects
  ↓
Final video assembly
```

### Agreed Technical Decisions

1. **The working input is mono or stereo.** Models do not receive the original
   six 5.1 channels. If the source contains only 5.1 audio, it is downmixed at
   the ingestion stage.
2. **16 kHz mono is the working format for VAD, diarization, and ASR.** A
   higher-quality 24/48 kHz copy may be stored separately for enhancement and
   voice cloning.
3. **The target order is enhancement before VAD.** This may help detect quiet
   speech under music. A control branch, `downmix → VAD`, is retained because an
   enhancement model may remove whispers, radio speech, or synthetic voices.
   The final decision is based on A/B metrics.
4. **Subtitle text is a draft reference, not absolute ground truth.** The test
   subset must be manually checked against the audio before WER/CER evaluation.
5. **DER cannot be calculated from ordinary subtitles alone.** It requires a
   manually verified RTTM file with exact intervals and speakers. Speaker labels
   from SDH subtitles may only be used as an annotation draft.
6. **Source videos and tracks are never modified.** All intermediate and final
   artifacts are written to separate directories.

## Data Contract Between the Groups

The first group passes a single utterance manifest to the second group:

| Field | Meaning |
|---|---|
| `episode` | Episode or video ID |
| `utterance_id` | Unique utterance ID |
| `start`, `end`, `duration` | Timing in seconds |
| `diarization_speaker` | Episode-local technical voice ID |
| `character_name` | Verified character name, when known |
| `text` | Verified source transcript |
| `style_hint` | Whisper, shout, radio, accent, or another delivery style |
| `text_needs_review` | Whether the transcript requires manual review |
| `speaker_needs_review` | Whether the speaker requires manual review |
| `speech_audio_path` | Path to the enhanced utterance audio |

The second group adds:

| Field | Meaning |
|---|---|
| `translated_text` | Translated utterance |
| `translation_status` | Automatic/manual review status |
| `voice_reference_id` | Character voice reference ID |
| `synthesized_audio_path` | Path to synthesized speech |
| `duration_error_ms` | Difference from the source duration |
| `tts_needs_review` | Whether synthesized speech requires review |

## Quality Metrics

### Original Content Analysis

- **WER** — the number of word substitutions, deletions, and insertions relative
  to a manually verified transcript;
- **CER** — the equivalent error rate at character level;
- **VAD Precision** — the proportion of detected intervals that actually
  contain speech;
- **VAD Recall** — the proportion of reference speech that was detected;
- **VAD F1** — the balance between precision and recall;
- **DER** — false alarm speech, missed speech, and speaker confusion divided by
  the total reference speech duration;
- **SER** — the proportion of speech time assigned to the wrong speaker. This
  exact definition of SER must be retained in all reports;
- **Character accuracy/coverage** — the proportion of correctly identified
  characters and the proportion of utterances for which a name is available.

A fixed test split that is never used for parameter tuning is required for fair
evaluation. The initial split must include clean dialogue, music, radio speech,
whispers, shouts, and overlapping speakers.

### Translation

- an automatic translation metric, such as COMET, on a verified subset;
- human evaluation of semantic accuracy and naturalness;
- the proportion of translations that fit the original timing without strong
  acceleration;
- absolute and relative utterance duration error.

### Voice Reproduction Quality

Voice cloning quality is evaluated with several complementary measures:

- **speaker similarity** — cosine similarity between embeddings of the source
  and synthesized voices;
- **MOS 1–5** — human evaluation of naturalness and overall quality;
- **similarity MOS 1–5** — human evaluation of resemblance to the character;
- **ASR WER/CER of synthesized speech** — intelligibility of the translated
  utterance;
- **F0/prosody** — similarity of pitch, pauses, energy, and emotional contour;
- **duration error** — deviation from the original time window;
- **clipping and loudness** — absence of overload and distracting loudness
  changes.

All TTS models must be evaluated on the same characters, utterances, and texts.
Human ratings should be collected from several listeners, not only from the
model developer.

## Checkpoint Plan

The dates below come from the academic project plan. Checkpoints 2–7 have
approximate deadlines.

### Checkpoint 1. Kickoff and Planning — October 6, 23:59

**Goal:** define the project scope, team, and a realistic annual plan.

**Tasks:**

- agree on the final project title and scope;
- create a team chat and add the supervisor;
- agree on the interface between the two groups;
- create the GitHub repository;
- document the topic, team, supervisor, architecture, metrics, and checkpoints;

**Deliverable:** repository, README files, responsibility assignment, and annual
plan. Every team member submits the repository link to the peer-review system
and sends it to the supervisor.

### Checkpoint 2. Exploratory Data Analysis — October 27, 23:59

**Goal:** understand the audio data and prepare a reproducible dataset.

**First group:**

- collect statistics on durations, channels, sample rates, and codecs;
- analyze loudness, silence, speech ratio, music, noise, and overlapping speech;
- study the quality and timing offset of English and Russian subtitles;
- visualize utterance durations, character counts, ASR confidence, and
  difficult scene types;
- define annotation rules for transcripts, VAD, and RTTM;
- create the fixed WER, CER, VAD, DER, and SER test set.

**Second group:**

- compare English and Russian text lengths;
- study translated speech duration relative to source timings;
- collect and inspect voice references;
- identify emotions and delivery styles important for TTS.

**Artifacts:** EDA notebook/report, tables and charts, dataset specification,
annotation guide, and a fixed test split.

### Checkpoint 3. First ML Models — November 27, 23:59

**Goal:** obtain a measurable baseline using ready-made models.

**First group:**

- compare one or two ready-made enhancement models;
- run a baseline VAD system;
- compare ASR models using WER/CER;
- run Nemotron and an alternative diarizer and measure DER/SER;
- run an A/B test of `enhancement → VAD` against `VAD without enhancement`;
- record configurations, runtime, and memory consumption.

**Second group:**

- select a translation baseline;
- test at least two ready-made TTS/voice-cloning models;
- implement text and audio fitting to the source timing;
- measure speaker similarity, MOS, synthesized-speech WER, and duration error.

**Deliverable:** initial end-to-end processing of several videos, a metrics
table, baseline error analysis, and a justified model selection.

### Checkpoint 4. ML Service — December 15, 23:59

**Goal:** expose the baseline through a service and a Telegram bot.

**FastAPI:**

- video upload and language selection;
- task creation and `job_id` response;
- status polling;
- result and report download;
- errors for size limits and unsupported formats.

**Telegram bot:**

- accepts a video or a file link;
- asks for the source and target language;
- displays the current processing stage and progress;
- returns the dubbed video;
- reports utterances that require manual review.

**Engineering requirements:** task queue, temporary storage, logging, `.env`
configuration, Docker, and basic automated tests.

**First-semester deliverable:** a working prototype based on ready-made models
and scripts, demonstrating the complete video-to-translated-video workflow.

### Midterm Project Presentation — January 10–15

- demonstrate the Telegram bot;
- present the architecture and each member's contribution;
- report baseline metrics for every pipeline stage;
- analyze successful and failed examples;
- define the second-semester fine-tuning plan.

### Checkpoint 5. ML Model Improvement — March 15, 23:59

**Goal:** improve the baseline through data, parameter tuning, and stronger
models.

- expand and clean the manual annotations;
- analyze VAD, ASR, diarization, translation, and TTS errors;
- tune thresholds and parameters only on the validation split;
- improve music/speech separation;
- improve global voice-cluster merging;
- increase robustness to whispers, radio speech, shouts, and background music;
- prepare data for prosody and voice-cloning fine-tuning;
- compare the improved pipeline with the baseline using fixed metrics.

### Checkpoint 6. Neural Networks — May 5, 23:59

**Goal:** fine-tune at least one key component and demonstrate a measurable
improvement.

The second-semester priority is prosody and speech synthesis quality, but the
final component is selected based on baseline error analysis.

Possible experiments:

- fine-tune a TTS/voice-cloning model on character voices and delivery styles;
- control emotion, F0, energy, and duration;
- adapt ASR to noisy and distorted voices;
- adapt speaker embeddings or the diarization component;
- train or fine-tune an enhancement model.

Every experiment records its data, architecture, hyperparameters, baseline,
final metrics, and representative errors.

### Checkpoint 7. Final Improvements

**Goal:** make experiments reproducible and prepare a stable demonstration.

- integrate MLflow;
- log parameters, metrics, model versions, and artifacts;
- retrain the best configuration on the fixed data;
- verify reproducibility from a clean environment;
- run load and robustness tests;
- measure runtime, memory use, and maximum supported video length;
- prepare final charts, tables, and an ablation study;
- document the API, bot, and local setup;
- document limitations and future work.

### Final Project Presentation — June 13–20

The final presentation demonstrates:

- a stable Telegram bot and the full end-to-end workflow;
- baseline versus improved system results;
- WER, CER, VAD Precision/Recall/F1, DER, and SER;
- translation and speech synthesis metrics;
- each member's contribution;
- error analysis, limitations, and robustness.

## Semester Plan

### First Semester: Ready-Made Models and a Working Application

- ready-made enhancement, VAD, ASR, and diarization models;
- a ready-made translation model;
- ready-made voice-cloning/TTS models;
- FastAPI service;
- Telegram bot;
- complete end-to-end baseline;
- fixed test set and initial metrics.

### Second Semester: Data, Fine-Tuning, and Quality

- expand the manual annotations;
- improve WER/CER/DER/SER;
- fine-tune the selected models;
- improve prosody and voice similarity;
- reduce duration error;
- add MLflow, reproducibility, and robustness analysis;
- optimize the final service.


## Definition of Done

The project is considered complete when:

1. The Telegram bot accepts a video and returns a dubbed result.
2. Every pipeline stage can be reproduced from a clean environment.
3. Metrics for every stage are reported on the fixed test split.
4. The improved system is compared with the first-semester baseline.
5. Utterances preserve the character, meaning, and an acceptable timing error.
6. Experiments and artifacts are registered in MLflow.
7. Known limitations are documented and supported with examples.
