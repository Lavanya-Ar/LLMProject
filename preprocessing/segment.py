#!/usr/bin/env python3
"""
Intelligent Lecture Transcript Segmentation
Splits lecture transcripts into logical sections based on content, topics, and structure
"""

import re
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Required libraries:
# pip install sentence-transformers scikit-learn nltk spacy
# python -m spacy download en_core_web_sm

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 Downloading NLTK 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab', quiet=True)


class LectureSegmenter:
    """
    Intelligently segments lecture transcripts using multiple strategies
    """
    
    def __init__(self, language='en'):
        """
        Initialize the segmenter with various NLP models
        
        Args:
            language: Language code ('en', 'fr', 'es', etc.)
        """
        print("🚀 Initializing Lecture Segmenter...")
        
        # Load sentence transformer for semantic similarity
        print("📥 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load spaCy for linguistic features
        print("📥 Loading spaCy model...")
        self.nlp = spacy.load('en_core_web_sm')
        
        # Define lecture-specific markers
        self.topic_markers = [
            r'so (next|now) (we\'?ll|let\'?s) (talk about|discuss|cover|move on to)',
            r'(moving|switching) (on|gears) to',
            r'let\'?s (now )?(consider|examine|look at|turn to)',
            r'the (next|following) (topic|subject|point|section) (is|will be)',
            r'(first|second|third|finally|lastly),? (we\'?ll|let\'?s)',
            r'(chapter|section|part) \d+',
            r'(now,? )?(what )?i want to (do now is|talk about|discuss|focus on)',
            r'(okay|ok|all right|alright),? (so )?(now )?(let\'?s|we\'?ll)',
        ]
        
        self.conclusion_markers = [
            r'(in )?conclusion',
            r'to (sum up|summarize|wrap up)',
            r'let\'?s (quickly )?recap',
            r'(that|this) (brings|wraps) us? to the end',
            r'(any )?(questions|comments)\?',
        ]
        
        print("✅ Segmenter ready!")
        
    def load_transcript(self, transcript_path, format_type='auto'):
        """
        Load transcript from various formats
        
        Args:
            transcript_path: Path to transcript file
            format_type: 'txt', 'srt', 'json', or 'auto'
            
        Returns:
            List of (text, timestamp) tuples
        """
        path = Path(transcript_path)
        
        if format_type == 'auto':
            if path.suffix == '.srt':
                format_type = 'srt'
            elif path.suffix == '.json':
                format_type = 'json'
            else:
                format_type = 'txt'
        
        print(f"📖 Loading {format_type.upper()} transcript from {path}")
        
        if format_type == 'txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [(text, 0)]
            
        elif format_type == 'srt':
            return self._parse_srt(path)
            
        elif format_type == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'segments' in data:
                return [(seg['text'], seg.get('start', i*10)) 
                       for i, seg in enumerate(data['segments'])]
            else:
                return [(data.get('full_text', ''), 0)]
        
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _parse_srt(self, srt_path):
        """Parse SRT subtitle file"""
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple SRT parser
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp line
                timestamp = lines[1]
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) -->', timestamp)
                if time_match:
                    start_time = self._srt_time_to_seconds(time_match.group(1))
                    text = ' '.join(lines[2:])
                    segments.append((text, start_time))
        
        return segments
    
    def _srt_time_to_seconds(self, srt_time):
        """Convert SRT time to seconds"""
        h, m, s = srt_time.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def preprocess_transcript(self, segments, min_sentence_length=5):
        """
        Preprocess transcript into sentences with context
        
        Args:
            segments: List of (text, timestamp) tuples
            min_sentence_length: Minimum words to keep sentence
            
        Returns:
            List of sentence dictionaries
        """
        print("\n🔧 Preprocessing transcript...")
        
        sentences = []
        current_time = 0
        
        for text, timestamp in segments:
            # Use provided timestamp or estimate
            if timestamp > 0:
                current_time = timestamp
            
            # Split into sentences
            raw_sentences = sent_tokenize(text)
            
            for sent in raw_sentences:
                sent = sent.strip()
                if len(sent.split()) >= min_sentence_length:
                    sentences.append({
                        'text': sent,
                        'start_time': current_time,
                        'words': len(sent.split()),
                        'processed': self._process_sentence(sent)
                    })
            
            # Rough time estimate (assuming 150 words per minute)
            current_time += len(text.split()) / 2.5  # Rough estimate
        
        print(f"✅ Found {len(sentences)} sentences")
        return sentences
    
    def _process_sentence(self, sentence):
        """Extract linguistic features from sentence"""
        doc = self.nlp(sentence)
        return {
            'has_question': any(token.text == '?' for token in doc),
            'has_transition': any(token.lemma_ in ['move', 'next', 'now', 'then'] 
                                  for token in doc),
            'nouns': [token.text for token in doc if token.pos_ == 'NOUN'],
            'entities': [(ent.text, ent.label_) for ent in doc.ents]
        }
    
    def detect_topic_boundaries(self, sentences, window_size=3, threshold=0.6):
        """
        Detect topic boundaries using semantic similarity
        
        Args:
            sentences: List of sentence dictionaries
            window_size: Number of sentences to consider for coherence
            threshold: Similarity threshold for boundary detection
            
        Returns:
            List of boundary indices
        """
        print("\n📊 Detecting topic boundaries...")
        
        # Get sentence embeddings
        texts = [s['text'] for s in sentences]
        embeddings = self.sentence_model.encode(texts)
        
        # Calculate coherence scores
        boundaries = []
        
        for i in range(window_size, len(sentences) - window_size):
            # Get windows before and after
            before_window = embeddings[i-window_size:i]
            after_window = embeddings[i:i+window_size]
            
            # Calculate average similarity within windows
            before_sim = cosine_similarity(before_window)
            after_sim = cosine_similarity(after_window)
            
            # Calculate cross-window similarity
            cross_sim = cosine_similarity(before_window, after_window)
            
            # Coherence score: higher when within-window similarity is high
            # and cross-window similarity is low
            within_coherence = (np.mean(before_sim) + np.mean(after_sim)) / 2
            cross_coherence = np.mean(cross_sim)
            
            # Boundary score
            boundary_score = within_coherence - cross_coherence
            
            if boundary_score < -threshold:  # Significant topic shift
                boundaries.append(i)
        
        print(f"📍 Found {len(boundaries)} potential topic boundaries")
        return boundaries
    
    def detect_section_markers(self, sentences):
        """
        Detect explicit section markers in the lecture
        
        Args:
            sentences: List of sentence dictionaries
            
        Returns:
            List of (index, marker_type) tuples
        """
        print("\n🔍 Detecting section markers...")
        
        markers = []
        
        for i, sent in enumerate(sentences):
            text = sent['text'].lower()
            
            # Check for topic introduction markers
            for pattern in self.topic_markers:
                if re.search(pattern, text, re.IGNORECASE):
                    markers.append((i, 'topic_start'))
                    break
            
            # Check for conclusion markers
            for pattern in self.conclusion_markers:
                if re.search(pattern, text, re.IGNORECASE):
                    markers.append((i, 'conclusion'))
                    break
            
            # Check for questions (often indicate Q&A sections)
            if sent['processed']['has_question']:
                markers.append((i, 'question'))
        
        print(f"📌 Found {len(markers)} section markers")
        return markers
    
    def extract_keywords(self, sentences, n_keywords=5):
        """
        Extract keywords for each potential section
        
        Args:
            sentences: List of sentence dictionaries
            n_keywords: Number of keywords per section
            
        Returns:
            List of keyword sets
        """
        # Group sentences into rough sections (every ~20 sentences)
        section_size = 20
        sections = []
        
        for i in range(0, len(sentences), section_size):
            section_text = ' '.join([s['text'] for s in sentences[i:i+section_size]])
            sections.append(section_text)
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sections)
        
        feature_names = vectorizer.get_feature_names_out()
        section_keywords = []
        
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray()[0]
            top_indices = row.argsort()[-n_keywords:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            section_keywords.append(keywords)
        
        return section_keywords
    
    def segment_lecture(self, sentences, method='hybrid'):
        """
        Main segmentation method combining multiple approaches
        
        Args:
            sentences: List of sentence dictionaries
            method: 'semantic', 'marker', or 'hybrid'
            
        Returns:
            List of section dictionaries
        """
        print(f"\n🎯 Segmenting lecture using {method} method...")
        
        # Get boundaries from different methods
        semantic_boundaries = self.detect_topic_boundaries(sentences)
        markers = self.detect_section_markers(sentences)
        marker_indices = [idx for idx, _ in markers]
        
        # Combine boundaries based on method
        if method == 'semantic':
            boundaries = sorted(set(semantic_boundaries))
        elif method == 'marker':
            boundaries = sorted(set(marker_indices))
        else:  # hybrid
            # Weighted combination
            all_boundaries = set(semantic_boundaries + marker_indices)
            boundaries = []
            
            for b in sorted(all_boundaries):
                weight = 1
                if b in semantic_boundaries:
                    weight += 2
                if b in marker_indices:
                    weight += 3
                boundaries.append((b, weight))
            
            # Keep only high-weight boundaries
            boundaries = [b for b, w in boundaries if w >= 3]
        
        # Get section keywords
        section_keywords = self.extract_keywords(sentences)
        
        # Create sections
        sections = []
        start_idx = 0
        
        for i, boundary in enumerate(boundaries):
            if boundary > start_idx:
                section_sentences = sentences[start_idx:boundary]
                section = self._create_section(
                    section_sentences,
                    i,
                    section_keywords[min(i, len(section_keywords)-1)]
                )
                sections.append(section)
                start_idx = boundary
        
        # Add final section
        if start_idx < len(sentences):
            section_sentences = sentences[start_idx:]
            section = self._create_section(
                section_sentences,
                len(sections),
                section_keywords[min(len(sections), len(section_keywords)-1)]
            )
            sections.append(section)
        
        print(f"✅ Created {len(sections)} logical sections")
        return sections
    
    def _create_section(self, sentences, index, keywords):
        """Create a section dictionary from sentences"""
        if not sentences:
            return None
        
        # Combine text
        text = ' '.join([s['text'] for s in sentences])
        
        # Get time range
        start_time = sentences[0]['start_time']
        end_time = sentences[-1]['start_time']
        
        # Generate title
        title = self._generate_title(sentences, keywords)
        
        # Count important terms
        nouns = []
        entities = []
        for s in sentences:
            nouns.extend(s['processed']['nouns'])
            entities.extend(s['processed']['entities'])
        
        # Get most common nouns (potential subtopics)
        noun_freq = defaultdict(int)
        for noun in nouns:
            noun_freq[noun] += 1
        top_nouns = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'section_number': index + 1,
            'title': title,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'text': text,
            'keywords': keywords,
            'key_nouns': [noun for noun, _ in top_nouns],
            'entities': list(set(entities))[:5],
            'sentence_count': len(sentences),
            'word_count': sum(s['words'] for s in sentences),
            'sentences': [s['text'] for s in sentences]
        }
    
    def _generate_title(self, sentences, keywords):
        """Generate a descriptive title for a section"""
        # Use first sentence if it's short enough
        first_sent = sentences[0]['text']
        if len(first_sent.split()) <= 15:
            return first_sent
        
        # Otherwise combine keywords
        if keywords:
            # Capitalize keywords and join
            title = ' & '.join(k.capitalize() for k in keywords[:3])
            return title
        
        # Fallback
        words = first_sent.split()[:8]
        return ' '.join(words) + '...'
    
    def export_sections(self, sections, output_format='all'):
        """
        Export sections in various formats
        
        Args:
            sections: List of section dictionaries
            output_format: 'txt', 'json', 'markdown', 'html', or 'all'
        """
        output_dir = Path('lecture_sections')
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Exporting sections to {output_dir}/")
        
        if output_format in ['txt', 'all']:
            self._export_as_txt(sections, output_dir)
        
        if output_format in ['json', 'all']:
            self._export_as_json(sections, output_dir)
        
        if output_format in ['markdown', 'all']:
            self._export_as_markdown(sections, output_dir)
        
        if output_format in ['html', 'all']:
            self._export_as_html(sections, output_dir)
    
    def _export_as_txt(self, sections, output_dir):
        """Export as plain text with separators"""
        path = output_dir / 'lecture_sections.txt'
        with open(path, 'w', encoding='utf-8') as f:
            for section in sections:
                f.write(f"\n{'='*60}\n")
                f.write(f"SECTION {section['section_number']}: {section['title']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Time: {timedelta(seconds=int(section['start_time']))} - "
                       f"{timedelta(seconds=int(section['end_time']))}\n")
                f.write(f"Keywords: {', '.join(section['keywords'])}\n")
                f.write(f"{'-'*40}\n")
                f.write(section['text'])
                f.write("\n\n")
        print(f"  ✓ Saved {path}")
    
    def _export_as_json(self, sections, output_dir):
        """Export as JSON with full data"""
        path = output_dir / 'lecture_sections.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {path}")
    
    def _export_as_markdown(self, sections, output_dir):
        """Export as Markdown with nice formatting"""
        path = output_dir / 'lecture_sections.md'
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Lecture Sections\n\n")
            
            # Table of contents
            f.write("## Table of Contents\n\n")
            for section in sections:
                time_str = str(timedelta(seconds=int(section['start_time'])))
                f.write(f"{section['section_number']}. [{section['title']}](#section-{section['section_number']}) "
                       f"*({time_str})*\n\n")
            
            f.write("---\n\n")
            
            # Sections
            for section in sections:
                f.write(f"<a name='section-{section['section_number']}'></a>\n")
                f.write(f"## Section {section['section_number']}: {section['title']}\n\n")
                
                start = timedelta(seconds=int(section['start_time']))
                end = timedelta(seconds=int(section['end_time']))
                f.write(f"**Time:** {start} – {end}  \n")
                f.write(f"**Duration:** {section['duration']:.1f} seconds  \n")
                f.write(f"**Keywords:** {', '.join(section['keywords'])}  \n\n")
                
                f.write("### Content\n\n")
                f.write(section['text'])
                f.write("\n\n---\n\n")
        print(f"  ✓ Saved {path}")
    
    def _export_as_html(self, sections, output_dir):
        """Export as HTML with navigation"""
        path = output_dir / 'lecture_sections.html'
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Lecture Sections</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .section { margin-bottom: 40px; padding: 20px; border-left: 4px solid #4CAF50; background: #f9f9f9; }
        .section-header { background: #4CAF50; color: white; padding: 10px; margin: -20px -20px 20px -20px; }
        .keywords { color: #666; font-style: italic; }
        .timestamp { font-family: monospace; background: #eee; padding: 2px 5px; }
        .toc { background: #f0f0f0; padding: 20px; margin-bottom: 30px; }
        .toc a { text-decoration: none; color: #333; }
        .toc a:hover { color: #4CAF50; }
        hr { border: none; border-top: 1px solid #ddd; margin: 30px 0; }
    </style>
</head>
<body>
    <h1>Lecture Sections</h1>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ol>
"""
        
        for section in sections:
            time_str = str(timedelta(seconds=int(section['start_time'])))
            html += f"            <li><a href='#section-{section['section_number']}'>{section['title']}</a> <span class='timestamp'>({time_str})</span></li>\n"
        
        html += """        </ol>
    </div>
    
    <hr>
"""
        
        for section in sections:
            html += f"""
    <div class='section' id='section-{section['section_number']}'>
        <div class='section-header'>
            <h2>Section {section['section_number']}: {section['title']}</h2>
        </div>
        <p><strong>Time:</strong> <span class='timestamp'>{timedelta(seconds=int(section['start_time']))}</span> – <span class='timestamp'>{timedelta(seconds=int(section['end_time']))}</span></p>
        <p><strong>Duration:</strong> {section['duration']:.1f} seconds</p>
        <p class='keywords'><strong>Keywords:</strong> {', '.join(section['keywords'])}</p>
        <h3>Content</h3>
        <p>{section['text']}</p>
    </div>
    <hr>
"""
        
        html += """
</body>
</html>"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  ✓ Saved {path}")


def main():
    """
    Main function to demonstrate the segmenter
    """
    # Configuration
    TRANSCRIPT_PATH = "transcription_output/lec4_transformers_transcript.txt"  # Change this!
    
    if not Path(TRANSCRIPT_PATH).exists():
        print(f"❌ Transcript not found: {TRANSCRIPT_PATH}")
        print("\nPlease provide a transcript file from your previous transcription.")
        print("The file can be:")
        print("  - JSON from the video_transcriber.py")
        print("  - SRT subtitle file")
        print("  - Plain text file")
        return
    
    # Initialize segmenter
    segmenter = LectureSegmenter(language='en')
    
    # Load transcript
    segments = segmenter.load_transcript(TRANSCRIPT_PATH)
    
    # Preprocess into sentences
    sentences = segmenter.preprocess_transcript(segments)
    
    # Segment using hybrid approach (best results)
    sections = segmenter.segment_lecture(sentences, method='hybrid')
    
    # Export in all formats
    segmenter.export_sections(sections, output_format='all')
    
    # Print summary
    print("\n" + "="*60)
    print("📋 SEGMENTATION SUMMARY")
    print("="*60)
    for section in sections[:3]:  # Show first 3 sections
        start = str(timedelta(seconds=int(section['start_time'])))
        print(f"\n[{start}] Section {section['section_number']}: {section['title']}")
        print(f"    Keywords: {', '.join(section['keywords'][:5])}")
        print(f"    Duration: {section['duration']:.1f}s, {section['sentence_count']} sentences")
    
    if len(sections) > 3:
        print(f"\n... and {len(sections)-3} more sections")
    
    print(f"\n✅ Complete! Check the 'lecture_sections' folder for all outputs.")


if __name__ == "__main__":
    main()