# src/data_preprocessing.py

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import os
import sys

@dataclass
class Document:
    url: str
    title: str
    content: str
    doc_type: str

@dataclass
class Chunk:
    chunk_id: str
    url: str
    title: str
    doc_type: str
    content: str
    chunk_index: int
    total_chunks: int

class HollowKnightDataProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_doc_type_from_title(self, title: str) -> str:
        """Extract document type from title for Hollow Knight content"""
        title_lower = title.lower()
        
        # Define category patterns for Hollow Knight content
        if any(keyword in title_lower for keyword in ['charm', 'charms']):
            return "Charm"
        elif any(keyword in title_lower for keyword in ['boss', 'bosses']):
            return "Boss"
        elif any(keyword in title_lower for keyword in ['enemy', 'enemies']):
            return "Enemy"
        elif any(keyword in title_lower for keyword in ['location', 'area', 'kingdom']):
            return "Location"
        elif any(keyword in title_lower for keyword in ['character', 'npc']):
            return "Character"
        elif any(keyword in title_lower for keyword in ['item', 'artifact', 'relic']):
            return "Item"
        elif any(keyword in title_lower for keyword in ['spell', 'ability', 'skill']):
            return "Ability"
        elif any(keyword in title_lower for keyword in ['quest', 'mission']):
            return "Quest"
        else:
            return "General"
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Hollow Knight wiki text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove wiki-specific artifacts
        artifacts = [
            r'\[\d+\]',  # Reference numbers
            r'\[edit\]',  # Edit markers
            r'\[citation needed\]',  # Citation needed
            r'Main article:.*?(?=\n|$)',  # Main article references
            r'See also:.*?(?=\n|$)',  # See also sections
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)\'\"]', '', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved regex"""
        # Split by sentence endings, but avoid splitting on abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, document: Document) -> List[Chunk]:
        """Create chunks from a document with overlap for Hollow Knight content"""
        chunks = []
        
        # Clean the content
        cleaned_content = self.clean_text(document.content)
        
        if not cleaned_content:
            return chunks
        
        # Split into sentences for better chunk boundaries
        sentences = self.split_into_sentences(cleaned_content)
        
        if not sentences:
            return chunks
        
        current_chunk = []    # Accumulated sentences
        current_length = 0    # Character count
        chunk_index = 0       # Tracking position
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Decision point: if adding this sentence exceeds chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                chunk_id = f"{document.title.replace(' ', '_')}_chunk_{chunk_index}"
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    url=document.url,
                    title=document.title,
                    doc_type=document.doc_type,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    total_chunks=0  # Will be updated later
                ))
                
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.get_overlap_sentence_count(current_chunk):]
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s) for s in overlap_sentences) + len(overlap_sentences) - 1
                
                # Add current sentence to new chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_id = f"{document.title.replace(' ', '_')}_chunk_{chunk_index}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                url=document.url,
                title=document.title,
                doc_type=document.doc_type,
                content=chunk_content,
                chunk_index=chunk_index,
                total_chunks=0
            ))
        
        # Update total chunks for each chunk
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        return chunks
    
    def get_overlap_sentence_count(self, sentences: List[str]) -> int:
        """Calculate how many sentences to use for overlap"""
        if len(sentences) <= 3:
            return 1
        elif len(sentences) <= 6:
            return 2
        else:
            return 3
    
    def load_hollow_knight_data(self, input_file: str) -> List[Document]:
        """Load Hollow Knight Fandom Wiki data from JSON file"""
        print(f"Loading Hollow Knight data from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Handle different possible JSON structures
        if 'pages' in data:
            # Structure with metadata and pages
            pages = data['pages']
        else:
            # Direct list of pages
            pages = data
        
        for item in pages:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            
            # Skip empty content
            if not content or len(content.strip()) < 50:
                continue
                
            doc_type = self.extract_doc_type_from_title(title)
            
            documents.append(Document(
                url=url,
                title=title,
                content=content,
                doc_type=doc_type
            ))
        
        print(f"Loaded {len(documents)} documents from Hollow Knight Wiki")
        return documents
    
    def process_documents(self, input_file: str, output_file: str) -> None:
        """Process all documents and save chunks to JSON"""
        print("Starting Hollow Knight Wiki data processing...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load documents
        documents = self.load_hollow_knight_data(input_file)
        
        if not documents:
            print("No documents found to process!")
            return
        
        # Process all documents and create chunks
        all_chunks = []
        chunk_stats = {}
        content_lengths = []
        
        for i, doc in enumerate(documents):
            if (i + 1) % 10 == 0:
                print(f"Processing document {i + 1}/{len(documents)}...")
                
            chunks = self.create_chunks(doc)
            all_chunks.extend(chunks)
            
            # Update statistics
            if doc.doc_type not in chunk_stats:
                chunk_stats[doc.doc_type] = 0
            chunk_stats[doc.doc_type] += len(chunks)
            
            # Collect content lengths for statistics
            for chunk in chunks:
                content_lengths.append(len(chunk.content))
        
        # Convert chunks to dictionaries for JSON serialization
        chunk_dicts = []
        for chunk in all_chunks:
            chunk_dicts.append({
                "chunk_id": chunk.chunk_id,
                "url": chunk.url,
                "title": chunk.title,
                "doc_type": chunk.doc_type,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "content_length": len(chunk.content)
            })
        
        # Save chunks to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"Saved chunks to: {output_file}")
        
        # Print detailed statistics
        self.print_statistics(chunk_stats, content_lengths, all_chunks, chunk_dicts)
    
    def print_statistics(self, chunk_stats, content_lengths, all_chunks, chunk_dicts):
        """Print detailed processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Total Documents Processed: {len(set(chunk.doc_type for chunk in all_chunks))}")
        print(f"Total Chunks Created: {len(all_chunks)}")
        
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            min_length = min(content_lengths)
            max_length = max(content_lengths)
            print(f"Average Chunk Size: {avg_length:.0f} characters")
            print(f"Chunk Size Range: {min_length} - {max_length} characters")
            
            # Size distribution
            small = len([l for l in content_lengths if l < 300])
            medium = len([l for l in content_lengths if 300 <= l < 600])
            large = len([l for l in content_lengths if l >= 600])
            print(f"Chunk Size Distribution:")
            print(f"  Small (<300): {small} chunks")
            print(f"  Medium (300-600): {medium} chunks")
            print(f"  Large (600+): {large} chunks")
        
        print(f"\nChunk distribution by document type:")
        for doc_type, count in sorted(chunk_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count} chunks")
        
        # Print some sample chunks
        print(f"\nSample chunks (first 3):")
        for i, chunk in enumerate(chunk_dicts[:3]):
            print(f"Chunk {i+1}:")
            print(f"  Title: {chunk['title']}")
            print(f"  Type: {chunk['doc_type']}")
            print(f"  ID: {chunk['chunk_id']}")
            print(f"  Content preview: {chunk['content'][:150]}...")
            print()

def main():
    """Main function to process Hollow Knight wiki data"""
    # Configuration for Hollow Knight data
    INPUT_FILE = "data/hollow_knight_fandom_corpus.json"
    OUTPUT_FILE = "data/hollow_knight_fandom_rag_optimized.json"
    
    # Initialize processor with optimized parameters for game content
    processor = HollowKnightDataProcessor(
        chunk_size=600,    # Slightly larger chunks for game content
        chunk_overlap=100  # More overlap for better context
    )
    
    # Process the data
    processor.process_documents(INPUT_FILE, OUTPUT_FILE)
    
    print(f"\n V Processing completed!")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()