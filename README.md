# LLMProject

This repo now has a working integration path for the project report goal:

`MP4 lecture upload -> transcription -> transcript segmentation -> concept extraction -> live MCQ generation -> interactive quiz UI`

## Main files

- `app.py`
  Streamlit frontend for upload, processing, quiz generation, answering, and scoring.
- `lecture_quiz_pipeline.py`
  Backend orchestration for saving uploaded videos and producing transcript, segments, and concepts.
- `quiz_generation.py`
  Live MCQ generation from extracted concepts and lecture segments.
- `preprocessing/transcribe.py`
  Local Whisper-based MP4 transcription pipeline.
- `segment_transcript.py`
  Root-level transcript segmentation and topic labeling used by the Streamlit app.
- `content_extraction.py`
  Root-level concept extraction used by the Streamlit app.

## How to run

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure `ffmpeg` is installed and available in your system PATH.

3. Create a `.env` file from `.env.example` and set:

- `LLM_PROVIDER`
- the matching API key for that provider

4. Start the app:

```bash
streamlit run app.py
```

## App flow

1. Upload an MP4 lecture recording.
2. Click `Process Lecture`.
3. The app saves artifacts into `runs/<timestamp>_<lecture-name>/`.
4. Review the transcript preview, detected topics, and extracted concepts.
5. Generate a quiz with the chosen topic, difficulty, and number of questions.
6. Attempt the quiz and review per-question explanations and final score.

## Output artifacts

Each run folder contains:

- uploaded video
- transcript `.txt`
- transcript `.json`
- subtitles `.srt`
- `segments.json`
- `concepts.json`
- `quiz_bank.json` after quiz generation

## Notes

- Lecture transcription is local and uses Whisper.
- Concept extraction and quiz generation use the configured LLM provider.
- If the LLM returns too few valid MCQs, the app can fill the remainder from the extracted concept bank so the demo still completes.
