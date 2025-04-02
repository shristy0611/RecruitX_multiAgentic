import re
from typing import List, Optional
import numpy as np # Import numpy


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into chunks of a target size with overlap, prioritizing meaningful separators.

    Args:
        text: The input text to split.
        chunk_size: The target maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # 1. Define separators (order matters - try more significant ones first)
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "] # Paragraphs, lines, sentences, spaces

    # 2. Initial split using the most significant separator
    final_chunks = []
    current_chunks = [text]

    for sep in separators:
        next_level_chunks = []
        for chunk in current_chunks:
            if len(chunk) <= chunk_size:
                next_level_chunks.append(chunk)
                continue

            # Split the chunk by the current separator
            if sep == " ": # Handle space splitting carefully
                 # Simple split by space if it's the last resort
                 sub_chunks = chunk.split(sep)
            else:
                 # Use regex to keep the separator at the end of the chunk (if possible)
                 # This helps maintain sentence structure better than simple split
                 try:
                     # Use lookbehind assertion to split *after* the separator
                     sub_chunks = re.split(f'(?<={re.escape(sep)})', chunk)
                 except re.error: # Handle potential regex errors (e.g., lookbehind limits)
                      sub_chunks = chunk.split(sep)
            
            # Filter out empty strings that might result from splitting
            sub_chunks = [sub for sub in sub_chunks if sub]
            next_level_chunks.extend(sub_chunks)
        
        current_chunks = next_level_chunks
        # Check if all chunks are now small enough after this separator
        if all(len(c) <= chunk_size for c in current_chunks):
            break # No need for finer-grained splitting
    
    # At this point, current_chunks contains pieces split by separators, 
    # but some might still be larger than chunk_size if separators were sparse.

    # 3. Merge small chunks and handle overlap / oversized chunks
    merged_chunks = []
    current_merged_chunk = ""
    
    for i, chunk in enumerate(current_chunks):
        # If a single chunk is already too large, split it forcefully
        if len(chunk) > chunk_size:
            # If we have something accumulated, store it first
            if current_merged_chunk:
                 merged_chunks.append(current_merged_chunk.strip())
                 current_merged_chunk = ""
            
            # Force split the large chunk
            for j in range(0, len(chunk), chunk_size - chunk_overlap):
                sub_chunk = chunk[j : j + chunk_size]
                merged_chunks.append(sub_chunk.strip())
            continue # Move to the next original chunk

        # Check if adding the next chunk exceeds the size limit
        if len(current_merged_chunk) + len(chunk) <= chunk_size:
            current_merged_chunk += chunk
        else:
            # Current merged chunk is full, store it
            if current_merged_chunk:
                merged_chunks.append(current_merged_chunk.strip())
            
            # Start the new chunk, considering overlap
            # Find the overlapping part from the *end* of the previous merged chunk
            overlap_start_index = max(0, len(current_merged_chunk) - chunk_overlap)
            overlap_text = current_merged_chunk[overlap_start_index:]
            
            # Start new chunk with overlap (if any) and the current small chunk
            current_merged_chunk = overlap_text.strip() + chunk 
            # If the new chunk itself is now too large after adding overlap (unlikely but possible)
            # handle it by just starting with the current small chunk
            if len(current_merged_chunk) > chunk_size:
                current_merged_chunk = chunk # Fallback

    # Add the last accumulated chunk
    if current_merged_chunk:
        merged_chunks.append(current_merged_chunk.strip())

    # Final filter for any empty strings that might have crept in
    final_chunks = [chunk for chunk in merged_chunks if chunk]

    return final_chunks 


def cosine_similarity(vec1: List[float], vec2: List[float]) -> Optional[float]:
    """Calculates the cosine similarity between two vectors."""
    try:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        if vec1_np.shape != vec2_np.shape or len(vec1_np.shape) != 1:
            # logger.warning("Vectors must have the same 1D shape for cosine similarity.")
            return None
            
        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            # logger.warning("Cannot compute cosine similarity with zero vector(s).")
            return 0.0 # Define similarity as 0 if one vector is zero
            
        similarity = dot_product / (norm_vec1 * norm_vec2)
        # Clip values to handle potential floating point inaccuracies slightly outside [-1, 1]
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception as e:
        # logger.error(f"Error calculating cosine similarity: {e}")
        return None 