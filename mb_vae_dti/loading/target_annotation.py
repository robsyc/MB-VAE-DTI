"""
Target annotation functionality for MB-VAE-DTI.

This module provides functions for annotating targets using their potential IDs
and AA sequences. It uses external APIs for retrieving target information.
"""

import requests
import time
from typing import Dict, Set, Optional, Any, Tuple, List
from pathlib import Path
import json
from pydantic import BaseModel
from Bio import Entrez, SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.Blast import NCBIWWW, NCBIXML
from io import StringIO
import re
import numpy as np
import hashlib

SEQ_SIMILARITY_THRESHOLD = 0.9  # Minimum similarity threshold for sequence matching
TOP_N_MATCHES = 3               # Number of top matches to check from API search results

# Set email for Entrez
Entrez.email = "robbe.claeys@ugent.be"  # Replace with your email

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = PROCESSED_DIR / "annotations"

# Ensure directories exist
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# Cache for API responses to avoid redundant calls
UNIPROT_CACHE_FILE = ANNOTATION_DIR / "uniprot_cache.json"
ENTREZ_CACHE_FILE = ANNOTATION_DIR / "entrez_cache.json"
BLAST_CACHE_FILE = ANNOTATION_DIR / "blast_cache.json"

# Initialize caches
uniprot_cache = {}
entrez_cache = {}
blast_cache = {}

# Load caches if they exist
if UNIPROT_CACHE_FILE.exists():
    try:
        with open(UNIPROT_CACHE_FILE, 'r') as f:
            uniprot_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load UniProt cache: {e}")

if ENTREZ_CACHE_FILE.exists():
    try:
        with open(ENTREZ_CACHE_FILE, 'r') as f:
            entrez_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load Entrez cache: {e}")

if BLAST_CACHE_FILE.exists():
    try:
        with open(BLAST_CACHE_FILE, 'r') as f:
            blast_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load BLAST cache: {e}")

def save_cache() -> None:
    """Save API response caches to disk."""
    try:        
        with open(UNIPROT_CACHE_FILE, 'w') as f:
            json.dump(uniprot_cache, f)
        
        with open(ENTREZ_CACHE_FILE, 'w') as f:
            json.dump(entrez_cache, f)
        
        with open(BLAST_CACHE_FILE, 'w') as f:
            json.dump(blast_cache, f)
    except Exception as e:
        print(f"Failed to save target cache: {e}")


class TargetAnnotation(BaseModel):
    """Target annotation model."""
    uniprot_id: Optional[str] = None
    aa_sequence: str
    gene_name: Optional[str] = None
    refseq_id: Optional[str] = None
    dna_sequence: Optional[str] = None
    similarity: float = 0.0
    valid: bool = False


def clean_sequence(sequence: str) -> str:
    """
    Clean a sequence by removing whitespace and non-standard characters.
    
    Args:
        sequence: Sequence to clean
        
    Returns:
        Cleaned sequence
    """
    # Remove whitespace
    sequence = sequence.strip()
    
    # Remove any non-standard characters
    sequence = ''.join(c for c in sequence if c.isalpha())
    
    return sequence


def clean_gene_name(gene_name: str) -> str:
    """
    Clean a gene name by applying various transformations to improve matching.
    
    Args:
        gene_name: Gene name to clean
        
    Returns:
        Cleaned gene name
    """
    import re
    
    # Make a copy of the original name
    cleaned_name = gene_name
    
    # Remove anything in parentheses
    cleaned_name = re.sub(r'\([^)]*\)', '', cleaned_name)
    
    # Replace hyphens with spaces
    cleaned_name = cleaned_name.replace('-', ' ')
    
    # Replace common Greek letters with their single-letter equivalents
    greek_replacements = {
        'alpha': 'A',
        'beta': 'B',
        'gamma': 'G',
        'delta': 'D',
        'epsilon': 'E',
        'zeta': 'Z',
        'eta': 'H',
        'theta': 'T',
        'iota': 'I',
        'kappa': 'K',
        'lambda': 'L',
        'mu': 'M',
        'nu': 'N',
        'xi': 'X',
        'omicron': 'O',
        'pi': 'P',
        'rho': 'R',
        'sigma': 'S',
        'tau': 'T',
        'upsilon': 'U',
        'phi': 'F',
        'chi': 'C',
        'psi': 'P',
        'omega': 'O'
    }
    
    for greek, replacement in greek_replacements.items():
        cleaned_name = re.sub(rf'\b{greek}\b', replacement, cleaned_name, flags=re.IGNORECASE)
    
    # Remove trailing 'p' (e.g., ABL1p -> ABL1)
    cleaned_name = re.sub(r'p$', '', cleaned_name)
    
    # Remove any remaining non-alphanumeric characters
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_name)
    
    # Trim whitespace and return
    return cleaned_name.strip()


def compare_sequences(seq1: str, seq2: str, threshold: float = SEQ_SIMILARITY_THRESHOLD) -> bool:
    """
    Compare two sequences and determine if they match.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        threshold: Minimum similarity threshold (default: SEQ_SIMILARITY_THRESHOLD)
        
    Returns:
        True if sequences match, False otherwise
    """
    # Clean sequences
    seq1 = clean_sequence(seq1)
    seq2 = clean_sequence(seq2)
    
    # If either sequence is empty, return False
    if not seq1 or not seq2:
        return False
    
    # If sequences are identical, return True
    if seq1 == seq2:
        return True
    
    # If sequences have different lengths, check if one is a substring of the other
    if len(seq1) != len(seq2):
        # Check if the shorter sequence is a substring of the longer one
        if len(seq1) < len(seq2):
            if seq1 in seq2:
                return True
        else:
            if seq2 in seq1:
                return True
    
    # Calculate similarity
    matches = sum(a == b for a, b in zip(seq1, seq2))
    similarity = matches / max(len(seq1), len(seq2))
    
    return similarity >= threshold


def translate_dna_to_aa(dna_sequence: str) -> str:
    """
    Translate a DNA sequence to amino acid sequence.
    
    Args:
        dna_sequence: DNA sequence to translate
        
    Returns:
        Amino acid sequence
    """
    try:
        # Clean the DNA sequence
        dna_sequence = clean_sequence(dna_sequence)
        
        # Create a Seq object and translate
        seq = Seq(dna_sequence)
        aa_sequence = str(seq.translate())
        
        # Remove stop codon if present
        if aa_sequence.endswith('*'):
            aa_sequence = aa_sequence[:-1]
            
        return aa_sequence
    except Exception as e:
        print(f"Error translating DNA sequence: {e}")
        return ""


class UniProtResult(BaseModel):
    primaryAccession: str           # e.g. "Q2M2I8"
    uniProtkbId: str                # e.g. "AAK1_HUMAN"
    sequence: str                   # e.g. "MALWMRLLPLLALLALWGPDPAAA..."
    geneName: Optional[str]         # e.g. "AAK1"
    RefSeq_id: Optional[str]        # e.g. "NM_014911"
    # EMBL_id: Optional[str]          # e.g. "AB028971"
    # potentially also CCDS_id

def search_uniprot_by_accession(uniprot_id: str) -> Optional[UniProtResult]:
    """
    Search UniProt by accession code.
    
    Args:
        uniprot_id: UniProt accession code to search for
        
    Returns:
        UniProtResult object with protein information or None if not found
    """
    # Check cache first
    if f"{uniprot_id}_acc" in uniprot_cache:
        return UniProtResult(**uniprot_cache[f"{uniprot_id}_acc"]) if uniprot_cache[f"{uniprot_id}_acc"] else None
    
    # UniProt API URL
    # e.g. https://rest.uniprot.org/uniprotkb/Q2M2I8.json
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url)
        
        # Respect rate limits
        time.sleep(0.2)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            EMBL_id = None
            refseq_id = None
            
            for cross_reference in data.get('uniProtKBCrossReferences', []):
                if EMBL_id and refseq_id:
                    break
                if cross_reference.get('database') == 'EMBL' and EMBL_id is None:
                    EMBL_id = cross_reference.get('id')
                elif cross_reference.get('database') == 'RefSeq' and refseq_id is None:
                    for property in cross_reference.get('properties', []):
                        if property.get('key') == 'NucleotideSequenceId':
                            refseq_id = property.get('value').split('.')[0]
                            continue

            result = UniProtResult(
                primaryAccession=data.get('primaryAccession', ''),
                uniProtkbId=data.get('uniProtkbId', ''),
                sequence=data.get('sequence', {}).get('value', ''),
                geneName=data.get('genes', [{}])[0].get('geneName', {}).get('value', None),
                RefSeq_id=refseq_id,
                # EMBL_id=EMBL_id
            )
            
            # Cache the result
            uniprot_cache[f"{uniprot_id}_acc"] = result.model_dump()
            save_cache()
            
            return result
        else:
            # Cache negative result
            uniprot_cache[f"{uniprot_id}_acc"] = None
            save_cache()
            return None
            
    except Exception as e:
        print(f"Error searching UniProt for {uniprot_id}: {e}")
        return None


def search_uniprot_by_gene(gene_id: str) -> Optional[UniProtResult]:
    """
    Search UniProt by gene symbol.
    
    Args:
        gene_id: UniProt gene symbol to search for
        
    Returns:
        UniProtResult object with protein information or None if not found
    """
    # Check cache first
    if f"{gene_id}_gene" in uniprot_cache:
        return UniProtResult(**uniprot_cache[f"{gene_id}_gene"]) if uniprot_cache[f"{gene_id}_gene"] else None
    
    # UniProt API URL for gene search
    # e.g. https://rest.uniprot.org/uniprotkb/search?query=gene:AAK1+AND+reviewed:true
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_id}+AND+reviewed:true"
    
    try:
        response = requests.get(url)
        
        # Respect rate limits
        time.sleep(0.2)
        
        if response.status_code == 200:
            results = response.json()
            
            # Check if we have results
            if results.get('results', []):
                data = results['results'][0]
                # Extract relevant information
                EMBL_id = None
                refseq_id = None
                
                for cross_reference in data.get('uniProtKBCrossReferences', []):
                    if EMBL_id and refseq_id:
                        break
                    if cross_reference.get('database') == 'EMBL' and EMBL_id is None:
                        EMBL_id = cross_reference.get('id')
                    elif cross_reference.get('database') == 'RefSeq' and refseq_id is None:
                        for property in cross_reference.get('properties', []):
                            if property.get('key') == 'NucleotideSequenceId':
                                refseq_id = property.get('value').split('.')[0]
                                continue

                result = UniProtResult(
                    primaryAccession=data.get('primaryAccession', ''),
                    uniProtkbId=data.get('uniProtkbId', ''),
                    sequence=data.get('sequence', {}).get('value', ''),
                    geneName=data.get('genes', [{}])[0].get('geneName', {}).get('value', None),
                    RefSeq_id=refseq_id,
                    EMBL_id=EMBL_id
                )
                
                # Cache the result
                uniprot_cache[f"{gene_id}_gene"] = result.model_dump()
                save_cache()
                
                return result
            else:
                # Cache negative result
                uniprot_cache[f"{gene_id}_gene"] = None
                save_cache()
                return None
        else:
            # Cache negative result
            uniprot_cache[f"{gene_id}_gene"] = None
            save_cache()
            return None
            
    except Exception as e:
        print(f"Error searching UniProt for gene {gene_id}: {e}")
        return None


def get_cds_from_refseq(refseq_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the coding sequence (CDS) from a RefSeq ID.
    
    Args:
        refseq_id: The RefSeq ID (e.g., "NM_014911")
        
    Returns:
        Dictionary containing CDS sequence and metadata
    """
    # Check cache first
    if refseq_id in entrez_cache:
        return entrez_cache[refseq_id]
    
    # Fetch the record from NCBI
    try:
        handle = Entrez.efetch(db="nucleotide", id=refseq_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        # Respect rate limits
        time.sleep(0.2)
        
        # Initialize result dictionary
        result = {
            "cds_sequences": [],
            "cds_translations": []
        }
        
        # Extract CDS features
        for feature in record.features:
            if feature.type == "CDS":
                # Get CDS sequence
                cds_sequence = feature.extract(record.seq)
                result["cds_sequences"].append(str(cds_sequence))
                
                # Get protein translation if available
                if "translation" in feature.qualifiers:
                    result["cds_translations"].append(feature.qualifiers["translation"][0])
        
        # Cache the result
        entrez_cache[refseq_id] = result
        save_cache()
        
        return result
    
    except Exception as e:
        print(f"Error fetching RefSeq record {refseq_id}: {str(e)}")
        # Cache negative result to avoid repeated failed requests
        entrez_cache[refseq_id] = None
        save_cache()
        return None


def search_by_blast(aa_sequence: str) -> Optional[str]:
    """
    Search for a protein using BLAST.
    
    Args:
        aa_sequence: Amino acid sequence to search for
        
    Returns:
        UniProt ID of best match or None if not found
    """
    # Clean the sequence
    aa_sequence = clean_sequence(aa_sequence)

    # Check cache first
    # Use a hash of the sequence as the key to avoid long keys
    sequence_hash = hashlib.md5(aa_sequence.encode()).hexdigest()
    if sequence_hash in blast_cache:
        return blast_cache[sequence_hash]
    
    try:      
        # Run BLAST search
        result_handle = NCBIWWW.qblast("blastp", "swissprot", aa_sequence)
        
        # Parse the results
        blast_record = NCBIXML.read(result_handle)
        
        # Check if we have any hits
        if blast_record.alignments:
            result = blast_record.alignments[0].accession
            
            # Cache the results
            blast_cache[sequence_hash] = result
            save_cache()
                
            return result
        
        # If we get here, we couldn't find a match
        blast_cache[sequence_hash] = None
        save_cache()
        return None
        
    except Exception as e:
        print(f"Error searching by BLAST: {e}")
        return None


def validate_dna_sequence(dna_sequence: str, aa_sequence: str, verbose: bool = False) -> Tuple[bool, float, str]:
    """
    Validate a DNA sequence by translating it and comparing to the amino acid sequence.
    Check all reading frames and find the best match.
    
    Args:
        dna_sequence: DNA sequence to validate
        aa_sequence: Amino acid sequence to compare against
        verbose: Whether to print additional information
        
    Returns:
        Tuple of (is_valid, similarity_score, best_matching_dna_segment)
    """
    aa_sequence = clean_sequence(aa_sequence)
    dna_sequence = clean_sequence(dna_sequence)
    
    best_score = 0
    best_frame = None
    best_dna_segment = None
    
    # Check all 6 reading frames (3 forward, 3 reverse)
    dna_seq = Seq(dna_sequence)
    frames = [
        dna_seq[0:],
        dna_seq[1:],
        dna_seq[2:],
        dna_seq.reverse_complement()[0:],
        dna_seq.reverse_complement()[1:],
        dna_seq.reverse_complement()[2:]
    ]
    
    for i, frame in enumerate(frames):
        # Translate the frame
        translated = str(frame.translate())

        # Use basic string matching to find best matching region
        if aa_sequence in translated and abs(len(aa_sequence) * 3 - len(frame)) < 3:
            best_score = np.inf
            best_frame = i
            best_dna_segment = str(frame)
            break
        
        # Use local alignment to find best matching region
        alignments = pairwise2.align.localms(
            translated, aa_sequence,
            2,    # match score
            -1,   # mismatch score
            -2,   # gap open penalty
            -0.5  # gap extension penalty
        )
        
        if alignments:
            best_alignment = alignments[0]
            score = best_alignment.score
            
            if score > best_score:
                best_score = score
                best_frame = i
                
                # Extract the matching DNA segment
                start = best_alignment.start
                end = best_alignment.end
                frame_dna = frames[i][start*3:end*3]
                best_dna_segment = str(frame_dna)
    
    # Calculate similarity as a percentage of the amino acid sequence length
    similarity = min(1, best_score / (2 * len(aa_sequence)))  # Normalize by max possible score
    
    if verbose:
        if best_frame is not None:
            print(f"Best match in reading frame {best_frame+1}")
            print(f"Similarity score: {similarity:.2f}")
        else:
            print("No significant match found in any reading frame")
    
    # Consider valid if similarity is above threshold
    is_valid = similarity >= SEQ_SIMILARITY_THRESHOLD  # Threshold for validation
    
    return is_valid, similarity, best_dna_segment or dna_sequence


def annotate_target(aa_sequence: str, potential_ids: Set[str], verbose: bool = False) -> TargetAnnotation:
    """
    Annotate a target with DNA sequence and IDs using a waterfall approach.
    
    Args:
        aa_sequence: Amino acid sequence
        potential_ids: Set of potential IDs (UniProt or gene IDs)
        verbose: Whether to print additional information
        
    Returns:
        TargetAnnotation object
    """
    # Clean the amino acid sequence
    clean_aa = clean_sequence(aa_sequence)
    
    # Initialize annotation with default values
    annotation = TargetAnnotation(
        aa_sequence=clean_aa,
        dna_sequence=None,
        uniprot_id=None,
        gene_name=None,
        refseq_id=None,
        similarity=0.0,
        valid=False
    )
    
    # Track the best DNA sequence match
    valid_dna_found = False
    best_dna_sequence = None
    best_similarity_score = 0.0
    
    # Function to check if we have all required information
    def has_all_required_info():
        return annotation.uniprot_id and annotation.gene_name and annotation.refseq_id and valid_dna_found
    
    # Function to update DNA information if better match is found
    def update_dna_info(dna_seq, similarity):
        nonlocal valid_dna_found, best_dna_sequence, best_similarity_score
        
        if similarity > best_similarity_score:
            best_dna_sequence = dna_seq
            best_similarity_score = similarity
            
            # Check if we've found a valid DNA sequence
            if similarity >= SEQ_SIMILARITY_THRESHOLD:
                valid_dna_found = True
                return True  # Valid DNA found
        
        return False  # No improvement or not valid
    
    # Process potential IDs if available
    if potential_ids:
        # Create a set of IDs to try, including cleaned versions
        ids_to_try = set()
        for potential_id in potential_ids:
            ids_to_try.add(potential_id)
            cleaned_id = clean_gene_name(potential_id)
            if cleaned_id != potential_id:
                ids_to_try.add(cleaned_id)
        
        for potential_id in ids_to_try:
            if verbose:
                print(f"Trying potential ID: {potential_id}")
            
            # Stop if we have all required information
            if has_all_required_info():
                break
            
            # Try as UniProt accession first
            if not has_all_required_info():
                result = search_uniprot_by_accession(potential_id)
                
                if result and compare_sequences(clean_aa, result.sequence): 
                    # Update annotation with protein info
                    annotation.uniprot_id = result.primaryAccession
                    
                    # Update gene ID if we don't have one or if this is from a direct UniProt match
                    if not annotation.gene_name and result.geneName:
                        annotation.gene_name = result.geneName
                    
                    if not annotation.refseq_id and result.RefSeq_id:
                        annotation.refseq_id = result.RefSeq_id
                    
                    # Try to get DNA sequence if we don't have a valid one yet
                    if not valid_dna_found:
                        refseq_result = get_cds_from_refseq(result.RefSeq_id)
                        if refseq_result:
                            for dna, aa in zip(refseq_result['cds_sequences'], refseq_result['cds_translations']):
                                is_valid, similarity, best_segment = validate_dna_sequence(
                                    dna, clean_aa, verbose
                                )
                                if update_dna_info(best_segment, similarity):
                                    annotation.refseq_id = result.RefSeq_id
                                    break
            
            # Try as gene ID if we haven't found all required information
            if not has_all_required_info():
                result = search_uniprot_by_gene(potential_id)
                
                if result and compare_sequences(clean_aa, result.sequence):
                    # Update UniProt ID if we don't have one
                    if not annotation.uniprot_id:
                        annotation.uniprot_id = result.primaryAccession
                    
                    # Update gene ID if we don't have one
                    if not annotation.gene_name:
                        annotation.gene_name = result.geneName
                    
                    if not annotation.refseq_id:
                        annotation.refseq_id = result.RefSeq_id
                    
                    # Try to get DNA sequence if we don't have a valid one yet
                    if not valid_dna_found:
                        refseq_result = get_cds_from_refseq(result.RefSeq_id)
                        if refseq_result:
                            for dna, aa in zip(refseq_result['cds_sequences'], refseq_result['cds_translations']):
                                is_valid, similarity, best_segment = validate_dna_sequence(
                                    dna, clean_aa, verbose
                                )
                                if update_dna_info(best_segment, similarity):
                                    annotation.uniprot_id = result.primaryAccession
                                    annotation.refseq_id = result.RefSeq_id
                                    break
                    
            # If we have all required information, break out of the potential_ids loop
            if has_all_required_info():
                break
    
    # If we haven't found all required information, try BLAST search
    if not has_all_required_info():
        if verbose:
            print("Falling back to BLAST search")
        
        BLAST_uniprot_id = search_by_blast(clean_aa)

        if BLAST_uniprot_id:
            annotation.uniprot_id = BLAST_uniprot_id
            
            result = search_uniprot_by_accession(BLAST_uniprot_id)

            if result:
                annotation.gene_name = result.geneName
                annotation.refseq_id = result.RefSeq_id
                
                refseq_result = get_cds_from_refseq(result.RefSeq_id)
                if refseq_result:
                    for dna, aa in zip(refseq_result['cds_sequences'], refseq_result['cds_translations']):
                        is_valid, similarity, best_segment = validate_dna_sequence(
                            dna, clean_aa, verbose
                        )
                        if update_dna_info(best_segment, similarity):
                            break

    # Only mark as valid if we have all three required components
    annotation.valid = annotation.uniprot_id and annotation.gene_name and annotation.refseq_id and valid_dna_found
    annotation.similarity = best_similarity_score
    annotation.dna_sequence = best_dna_sequence

    if verbose and not annotation.valid:
        print(f"Failed to annotate target {aa_sequence}")
        print(f"Potential IDs: {potential_ids}")
    return annotation

