#!/usr/bin/env python3
"""
Complete Video Transcription Pipeline with Audio Chunking
Transcribes long MP4 videos by splitting audio into manageable chunks
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# Required libraries - install with:
# pip install openai-whisper pydub torch ffmpeg-python

import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch


class VideoTranscriber:
    """
    A complete pipeline for transcribing long videos with audio chunking
    """
    
    def __init__(self, model_size="base", device=None, language=None):
        """
        Initialize the transcriber
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: "cuda" for GPU, "cpu" for CPU (auto-detects if None)
            language: Language code (e.g., "en" for English, None for auto-detect)
        """
        print(f"🚀 Initializing VideoTranscriber with model: {model_size}")
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"💻 Using device: {self.device}")
        
        # Load Whisper model
        print(f"📥 Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size, device=self.device)
        self.language = language
        print(f"✅ Model loaded successfully!")
        
        # Create output directory
        self.output_dir = Path("transcription_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_audio_from_video(self, video_path):
        """
        Step 1: Extract audio from video file
        
        Args:
            video_path: Path to input MP4 video
            
        Returns:
            Path to extracted WAV audio file
        """
        print(f"\n🎬 Step 1: Extracting audio from {video_path}")
        
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Create audio filename in output directory
        audio_filename = self.output_dir / f"{video_file.stem}_audio.wav"
        
        try:
            # Load video and extract audio
            print("⏳ Loading video and extracting audio stream...")
            audio = AudioSegment.from_file(video_path, format="mp4")
            
            # Convert to mono and set sample rate for better compatibility
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export as WAV
            print(f"💾 Saving audio to {audio_filename}")
            audio.export(audio_filename, format="wav")
            
            # Get audio duration
            duration_seconds = len(audio) / 1000.0
            duration_str = str(timedelta(seconds=int(duration_seconds)))
            print(f"⏱️  Audio duration: {duration_str}")
            
            return str(audio_filename)
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            raise
    
    def split_audio_into_chunks(self, audio_path, chunk_length_ms=600000, min_silence_len=500):
        """
        Step 2: Split audio into manageable chunks
        
        Args:
            audio_path: Path to audio file
            chunk_length_ms: Maximum chunk length in milliseconds (default: 10 minutes)
            min_silence_len: Minimum silence length for splitting in milliseconds
            
        Returns:
            List of (chunk_path, start_time_ms) tuples
        """
        print(f"\n🔪 Step 2: Splitting audio into chunks (max {chunk_length_ms/60000:.0f} minutes each)")
        
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        
        # First, try to split on silence for natural breaks
        print("⏳ Detecting silence for natural splitting...")
        silence_chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=-40,  # dB, adjust based on your audio
            keep_silence=500  # Keep 500ms of silence at boundaries
        )
        
        print(f"📍 Found {len(silence_chunks)} natural segments")
        
        # If we have too many small chunks, merge them
        if len(silence_chunks) > 1:
            merged_chunks = []
            current_chunk = AudioSegment.empty()
            current_start = 0
            
            for chunk in silence_chunks:
                if len(current_chunk) + len(chunk) <= chunk_length_ms:
                    # Add to current chunk
                    if len(current_chunk) == 0:
                        current_start = sum(len(c) for c in merged_chunks)
                    current_chunk += chunk
                else:
                    # Save current chunk if not empty
                    if len(current_chunk) > 0:
                        merged_chunks.append((current_chunk, current_start))
                    # Start new chunk
                    current_chunk = chunk
                    current_start = sum(len(c) for c in merged_chunks) + len(current_chunk)
            
            # Add the last chunk
            if len(current_chunk) > 0:
                merged_chunks.append((current_chunk, current_start))
            
            chunks = merged_chunks
        else:
            # If no good silence breaks, just split by time
            print("⚠️  No natural breaks found, splitting by time...")
            for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
                end_ms = min(start_ms + chunk_length_ms, len(audio))
                chunk = audio[start_ms:end_ms]
                chunks.append((chunk, start_ms))
        
        print(f"📦 Created {len(chunks)} chunks for processing")
        
        # Save chunks to files
        chunk_paths = []
        audio_path_obj = Path(audio_path)
        
        for i, (chunk, start_ms) in enumerate(chunks):
            chunk_filename = self.output_dir / f"{audio_path_obj.stem}_chunk_{i:03d}.wav"
            chunk.export(chunk_filename, format="wav")
            chunk_paths.append((str(chunk_filename), start_ms))
            print(f"  Chunk {i+1}: {chunk_filename.name} (starts at {timedelta(seconds=start_ms/1000)})")
        
        return chunk_paths
    
    def transcribe_chunk(self, chunk_path, start_time_ms):
        """
        Step 3: Transcribe a single audio chunk
        
        Args:
            chunk_path: Path to chunk audio file
            start_time_ms: Start time in milliseconds
            
        Returns:
            Dictionary with transcript and metadata
        """
        print(f"🎤 Transcribing chunk: {Path(chunk_path).name}")
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                chunk_path,
                language=self.language,
                verbose=False,
                fp16=(self.device == "cuda")
            )
            
            # Add timestamps
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": start_time_ms/1000 + segment["start"],
                    "end": start_time_ms/1000 + segment["end"],
                    "text": segment["text"].strip()
                })
            
            print(f"  ✓ Found {len(segments)} segments")
            return {
                "text": result["text"],
                "segments": segments,
                "chunk_start": start_time_ms / 1000
            }
            
        except Exception as e:
            print(f"  ❌ Error transcribing chunk: {e}")
            return {
                "text": "",
                "segments": [],
                "chunk_start": start_time_ms / 1000,
                "error": str(e)
            }
    
    def transcribe_video(self, video_path, max_chunk_minutes=10):
        """
        Main pipeline: Transcribe entire video with chunking
        
        Args:
            video_path: Path to input MP4 video
            max_chunk_minutes: Maximum chunk length in minutes
            
        Returns:
            Dictionary with complete transcript and metadata
        """
        start_time = time.time()
        video_name = Path(video_path).stem
        
        print("\n" + "="*60)
        print(f"🎥 STARTING TRANSCRIPTION PIPELINE")
        print(f"📁 Video: {video_path}")
        print(f"⏱️  Max chunk size: {max_chunk_minutes} minutes")
        print("="*60)
        
        try:
            # Step 1: Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            
            # Step 2: Split into chunks
            chunk_length_ms = max_chunk_minutes * 60 * 1000
            chunks = self.split_audio_into_chunks(audio_path, chunk_length_ms=chunk_length_ms)
            
            # Step 3: Transcribe each chunk
            print(f"\n🔊 Step 3: Transcribing {len(chunks)} chunks...")
            all_segments = []
            full_text = []
            
            for i, (chunk_path, start_ms) in enumerate(chunks, 1):
                print(f"\n--- Chunk {i}/{len(chunks)} ---")
                chunk_result = self.transcribe_chunk(chunk_path, start_ms)
                
                if chunk_result["text"]:
                    full_text.append(chunk_result["text"])
                    all_segments.extend(chunk_result["segments"])
                
                # Clean up chunk file (optional - comment out to keep chunks)
                # os.remove(chunk_path)
            
            # Clean up main audio file (optional - comment out to keep audio)
            # os.remove(audio_path)
            
            # Step 4: Combine and format results
            print(f"\n📊 Step 4: Finalizing transcript...")
            
            # Sort segments by start time (just to be safe)
            all_segments.sort(key=lambda x: x["start"])
            
            # Calculate statistics
            total_duration = all_segments[-1]["end"] - all_segments[0]["start"] if all_segments else 0
            word_count = sum(len(segment["text"].split()) for segment in all_segments)
            
            # Create result object
            result = {
                "video_name": video_name,
                "full_text": " ".join(full_text),
                "segments": all_segments,
                "statistics": {
                    "total_segments": len(all_segments),
                    "total_duration_seconds": total_duration,
                    "total_duration": str(timedelta(seconds=int(total_duration))),
                    "word_count": word_count,
                    "processing_time": time.time() - start_time
                }
            }
            
            # Save results
            self.save_results(result, video_name)
            
            print("\n" + "="*60)
            print("✅ TRANSCRIPTION COMPLETE!")
            print(f"⏱️  Processing time: {result['statistics']['processing_time']:.2f} seconds")
            print(f"📝 Total words: {word_count}")
            print(f"💾 Results saved in: {self.output_dir}")
            print("="*60)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
    
    def save_results(self, result, video_name):
        """
        Save transcript in multiple formats
        """
        # Save plain text
        txt_path = self.output_dir / f"{video_name}_transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["full_text"])
        
        # Save SRT subtitles
        srt_path = self.output_dir / f"{video_name}_subtitles.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start = self._seconds_to_srt_time(segment["start"])
                end = self._seconds_to_srt_time(segment["end"])
                f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")
        
        # Save JSON with full metadata
        json_path = self.output_dir / f"{video_name}_transcript.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save timestamped text
        ts_path = self.output_dir / f"{video_name}_timestamps.txt"
        with open(ts_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = str(timedelta(seconds=int(segment["start"])))
                f.write(f"[{start}] {segment['text']}\n")
    
    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    """
    Main function to run the transcriber
    """
    # Configuration
    VIDEO_PATH = "your_video.mp4" 
    
    # Check if video path is provided
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found: {VIDEO_PATH}")
        print("\nUsage:")
        print("  python video_transcriber.py path/to/your/video.mp4")
        print("\nOr edit the VIDEO_PATH variable in the script.")
        sys.exit(1)
    
    # Model options (trade-off between speed and accuracy):
    # - "tiny": Fastest, least accurate
    # - "base": Good balance for most use cases
    # - "small": Better accuracy, slower
    # - "medium": High accuracy, much slower
    # - "large": Best accuracy, slowest
    MODEL_SIZE = "base"  # Change this based on your needs
    
    # Language options:
    # - None: Auto-detect
    # - "en": English only (faster)
    # - "fr": French
    # - "es": Spanish
    # - etc.
    LANGUAGE = "en"  # Set to None for auto-detect
    
    # Chunk size in minutes (adjust based on your RAM)
    # Smaller chunks use less memory but may have more context breaks
    CHUNK_SIZE_MINUTES = 10
    
    try:
        # Initialize transcriber
        transcriber = VideoTranscriber(
            model_size=MODEL_SIZE,
            language=LANGUAGE
        )
        
        # Run transcription
        result = transcriber.transcribe_video(
            VIDEO_PATH,
            max_chunk_minutes=CHUNK_SIZE_MINUTES
        )
        
        # Print a preview
        print("\n📋 Transcript Preview (first 500 chars):")
        print("-" * 40)
        print(result["full_text"][:500] + "...")
        print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Transcription interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()