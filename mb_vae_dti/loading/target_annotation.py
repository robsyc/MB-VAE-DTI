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
    aa_sequence: str
    dna_sequence: Optional[str] = None
    uniprot_id: Optional[str] = None
    gene_id: Optional[str] = None
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
            return seq1 in seq2
        else:
            return seq2 in seq1
    
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


def search_uniprot_by_id(uniprot_id: str) -> Optional[Dict[str, Any]]:
    """
    Search UniProt by ID.
    
    Args:
        uniprot_id: UniProt ID to search for
        
    Returns:
        Dictionary with protein information or None if not found
    """
    # Check cache first
    if uniprot_id in uniprot_cache:
        return uniprot_cache[uniprot_id]
    
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
            result = {
                'uniprot_id': uniprot_id,
                'aa_sequence': data.get('sequence', {}).get('value', ''),
                'gene_id': None
            }
            
            # Try to get gene ID
            for gene in data.get('genes', []):
                if gene.get('geneName', {}).get('value'):
                    result['gene_id'] = gene.get('geneName', {}).get('value')
                    break
            
            # Cache the result
            uniprot_cache[uniprot_id] = result
            save_cache()
            
            return result
        else:
            # Cache negative result
            uniprot_cache[uniprot_id] = None
            save_cache()
            return None
            
    except Exception as e:
        print(f"Error searching UniProt for {uniprot_id}: {e}")
        return None


def search_uniprot_by_gene(gene_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search UniProt by gene ID.
    
    Args:
        gene_id: Gene ID to search for
        
    Returns:
        List of dictionaries with protein information or None if not found
    """
    # Check cache first
    cache_key = f"gene_{gene_id}"
    if cache_key in uniprot_cache:
        return uniprot_cache[cache_key]
    
    # UniProt API URL for gene search
    # e.g. https://rest.uniprot.org/uniprotkb/search?query=gene:AAK1+AND+reviewed:true
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_id}+AND+reviewed:true"
    
    try:
        response = requests.get(url)
        
        # Respect rate limits
        time.sleep(0.2)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have results
            if data.get('results', []):
                # Get up to TOP_N_MATCHES results
                results = []
                for entry in data['results'][:TOP_N_MATCHES]:
                    # Extract relevant information
                    result = {
                        'uniprot_id': entry.get('primaryAccession'),
                        'aa_sequence': entry.get('sequence', {}).get('value', ''),
                        'gene_id': gene_id
                    }
                    results.append(result)
                
                # Cache the results
                uniprot_cache[cache_key] = results
                save_cache()
                
                return results
            else:
                # Cache negative result
                uniprot_cache[cache_key] = None
                save_cache()
                return None
        else:
            # Cache negative result
            uniprot_cache[cache_key] = None
            save_cache()
            return None
            
    except Exception as e:
        print(f"Error searching UniProt for gene {gene_id}: {e}")
        return None


def get_gene_sequence_from_entrez(gene_id: str) -> Optional[Dict[str, Any]]:
    """
    Get gene sequence information from NCBI Entrez.
    
    Args:
        gene_id: Gene ID to search for
        
    Returns:
        Dictionary with gene information or None if not found
    """
    # Check cache first
    if gene_id in entrez_cache:
        return entrez_cache[gene_id]
    
    try: 
        # First, search for the gene
        handle = Entrez.esearch(db="gene", term=f"{gene_id}[Gene Name] AND human[Organism]")
        record = Entrez.read(handle)
        handle.close()
        
        if record["Count"] == "0":
            return None
        
        # Get the gene ID
        gene_id_num = record["IdList"][0]
        
        # Get gene details
        handle = Entrez.efetch(db="gene", id=gene_id_num, retmode="xml")
        gene_record = Entrez.read(handle)
        handle.close()
        
        # Extract information
        gene_info = gene_record[0]
        
        # Get the mRNA/CDS sequences
        for genomic_info in gene_info.get("Entrezgene_locus", []):
            for ref in genomic_info.get("Gene-commentary_products", []):
                if ref.get("Gene-commentary_type") == 3:  # mRNA
                    # Get the accession
                    accession = ref.get("Gene-commentary_accession")
                    
                    # Fetch the mRNA record
                    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb")
                    gb_record = SeqIO.read(handle, "genbank")
                    handle.close()
                    
                    # Find CDS features
                    for feature in gb_record.features:
                        if feature.type == "CDS":
                            # Extract the CDS sequence
                            cds = str(feature.extract(gb_record.seq))
                            
                            result = {
                                "gene_id": gene_id,
                                "dna_sequence": cds,
                                "protein_id": feature.qualifiers.get("protein_id", [""])[0]
                            }
                            
                            # Cache the result
                            entrez_cache[gene_id] = result
                            save_cache()
                            
                            return result
    
    except Exception as e:
        print(f"Error fetching gene from Entrez: {e}")
    
    return None


def search_by_blast(aa_sequence: str) -> Optional[List[Dict[str, Any]]]:
    """
    Search for a protein using BLAST.
    
    Args:
        aa_sequence: Amino acid sequence to search for
        
    Returns:
        List of dictionaries with protein information or None if not found
    """
    # Check cache first
    # Use a hash of the sequence as the key to avoid long keys
    import hashlib
    sequence_hash = hashlib.md5(aa_sequence.encode()).hexdigest()
    
    if sequence_hash in blast_cache:
        return blast_cache[sequence_hash]
    
    try:
        # Clean the sequence
        aa_sequence = clean_sequence(aa_sequence)
        
        # Run BLAST search
        result_handle = NCBIWWW.qblast("blastp", "swissprot", aa_sequence)
        
        # Parse the results
        blast_record = NCBIXML.read(result_handle)
        
        # Check if we have any hits
        if blast_record.alignments:
            # Get up to TOP_N_MATCHES hits
            results = []
            for alignment in blast_record.alignments[:TOP_N_MATCHES]:
                hsp = alignment.hsps[0]
                
                # Extract the UniProt ID
                uniprot_id = alignment.accession
                
                # Get the full protein information from UniProt
                protein_info = search_uniprot_by_id(uniprot_id)
                
                if protein_info:
                    results.append(protein_info)
                    
                    # If we have enough results, stop
                    if len(results) >= TOP_N_MATCHES:
                        break
            
            if results:
                # Cache the results
                blast_cache[sequence_hash] = results
                save_cache()
                
                return results
        
        # If we get here, we couldn't find a match
        blast_cache[sequence_hash] = None
        save_cache()
        return None
        
    except Exception as e:
        print(f"Error searching by BLAST: {e}")
        return None


def get_dna_sequence_from_uniprot(uniprot_id: str) -> Optional[str]:
    """
    Get DNA sequence from UniProt API.
    
    Args:
        uniprot_id: UniProt ID to search for
        
    Returns:
        DNA sequence or None if not found
    """
    # Check cache first
    cache_key = f"dna_{uniprot_id}"
    if cache_key in uniprot_cache:
        return uniprot_cache[cache_key]
    
    try:
        # Use UniProt API to get CDS specifically
        # e.g. https://rest.uniprot.org/uniprotkb/Q2M2I8.xml
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.xml"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Extract CDS from XML
            xml_data = response.text
            record = SeqIO.read(StringIO(xml_data), "uniprot-xml")
            
            # Look for cross-references to EMBL/GenBank/DDBJ
            cds = None
            for dbref in record.dbxrefs:
                if dbref.startswith("EMBL:") or dbref.startswith("GenBank:") or dbref.startswith("DDBJ:"):
                    # Get the accession number
                    acc = dbref.split(":", 1)[1]
                    
                    # Use Entrez to get the CDS
                    handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb")
                    gb_record = SeqIO.read(handle, "genbank")
                    
                    # Find CDS features
                    for feature in gb_record.features:
                        if feature.type == "CDS":
                            if "protein_id" in feature.qualifiers:
                                # Extract the CDS sequence
                                cds = str(feature.extract(gb_record.seq))
                                break
                    
                    if cds:
                        break
            
            # Cache the result using a different key to avoid conflicts
            uniprot_cache[cache_key] = cds
            save_cache()
            
            return cds
        else:
            # Cache negative result
            uniprot_cache[cache_key] = None
            save_cache()
            return None
    
    except Exception as e:
        print(f"Error fetching CDS from UniProt: {e}")
    
    return None


def get_dna_sequence_by_gene_symbol(gene_symbol: str) -> Optional[str]:
    """
    Get DNA sequence by gene symbol from NCBI Nucleotide database.
    
    Args:
        gene_symbol: Gene symbol to search for
        
    Returns:
        DNA sequence or None if not found
    """
    # Check cache first
    cache_key = f"nucl_{gene_symbol}"
    if cache_key in entrez_cache:
        return entrez_cache[cache_key]
    
    try:
        # Search for the gene in nucleotide database
        handle = Entrez.esearch(db="nucleotide", term=f"{gene_symbol}[Gene Name] AND RefSeq[Filter] AND mRNA[Filter] AND human[Organism]")
        record = Entrez.read(handle)
        handle.close()
        
        # Respect rate limits
        time.sleep(1)
        
        # Check if we found any results
        if record["Count"] == "0":
            # Try a broader search
            handle = Entrez.esearch(db="nucleotide", term=f"{gene_symbol}[Gene Name] AND human[Organism]")
            record = Entrez.read(handle)
            handle.close()
            
            # Respect rate limits
            time.sleep(1)
            
            if record["Count"] == "0":
                # Cache negative result
                entrez_cache[cache_key] = None
                save_cache()
                return None
        
        # Get the first nucleotide ID
        nucleotide_id = record["IdList"][0]
        
        # Get the nucleotide sequence
        handle = Entrez.efetch(db="nucleotide", id=nucleotide_id, rettype="fasta", retmode="text")
        seq_record = SeqIO.read(handle, "fasta")
        handle.close()
        
        # Respect rate limits
        time.sleep(1)
        
        # Extract sequence
        dna_sequence = str(seq_record.seq)
        
        # Cache the result
        entrez_cache[cache_key] = dna_sequence
        save_cache()
        
        return dna_sequence
        
    except Exception as e:
        print(f"Error searching nucleotide database for gene {gene_symbol}: {e}")
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
        gene_id=None,
        valid=False
    )
    
    # Track the best DNA sequence match
    best_dna_sequence = None
    best_similarity_score = 0.0
    valid_dna_threshold = SEQ_SIMILARITY_THRESHOLD  # Threshold for considering a DNA sequence valid
    valid_dna_found = False
    
    # Track whether we've found a valid protein/gene match
    found_valid_match = False
    
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
            
            # Skip further DNA sequence searches if we already have a valid one
            skip_dna_search = valid_dna_found
            
            # Try as UniProt ID first
            if not skip_dna_search:
                protein_info = search_uniprot_by_id(potential_id)
                print(f"Protein info: {protein_info}")
                
                if protein_info and compare_sequences(clean_aa, protein_info['aa_sequence']):
                    if verbose:
                        print(f"Found match in UniProt for ID: {potential_id}")
                    
                    # Update annotation with protein info
                    annotation.uniprot_id = protein_info['uniprot_id']
                    annotation.gene_id = protein_info['gene_id']
                    annotation.valid = True
                    found_valid_match = True
                    
                    # Try to get DNA sequence if we don't have a valid one yet
                    if not valid_dna_found:
                        dna_sequence = get_dna_sequence_from_uniprot(protein_info['uniprot_id'])
                        
                        if dna_sequence:
                            # Validate the DNA sequence
                            is_valid, similarity, best_segment = validate_dna_sequence(
                                dna_sequence, clean_aa, verbose
                            )
                            
                            # Update best DNA sequence if this one is better
                            if similarity > best_similarity_score:
                                best_similarity_score = similarity
                                best_dna_sequence = best_segment
                                
                                # Check if we've found a valid DNA sequence
                                if similarity >= valid_dna_threshold:
                                    valid_dna_found = True
                                    if verbose:
                                        print(f"Found valid DNA sequence with similarity {similarity:.2f}")
            
            # Try as gene ID if we haven't found a valid DNA sequence yet
            if not valid_dna_found:
                gene_results = search_uniprot_by_gene(potential_id)
                print(f"Gene results: {gene_results}")
                
                if gene_results:
                    for gene_info in gene_results:
                        if compare_sequences(clean_aa, gene_info['aa_sequence']):
                            if verbose:
                                print(f"Found match in UniProt for gene: {potential_id}")
                            
                            # Update annotation with gene info if we don't have valid info yet
                            if not found_valid_match:
                                annotation.uniprot_id = gene_info['uniprot_id']
                                annotation.gene_id = potential_id
                                annotation.valid = True
                                found_valid_match = True
                            
                            # Try to get DNA sequence from Entrez
                            entrez_info = get_gene_sequence_from_entrez(potential_id)
                            print(f"Entrez info: {entrez_info}")
                            if entrez_info:
                                dna_sequence = entrez_info['dna_sequence']
                                
                                if dna_sequence:
                                    # Validate the DNA sequence
                                    is_valid, similarity, best_segment = validate_dna_sequence(
                                        dna_sequence, clean_aa, verbose
                                    )
                                    
                                    # Update best DNA sequence if this one is better
                                    if similarity > best_similarity_score:
                                        best_similarity_score = similarity
                                        best_dna_sequence = best_segment
                                        
                                        # Check if we've found a valid DNA sequence
                                        if similarity >= valid_dna_threshold:
                                            valid_dna_found = True
                                            if verbose:
                                                print(f"Found valid DNA sequence with similarity {similarity:.2f}")
                                            break  # Break out of the gene_results loop
                            
                            # If we still don't have a valid DNA sequence, try by gene symbol
                            if not valid_dna_found:
                                dna_sequence = get_dna_sequence_by_gene_symbol(potential_id)
                                print(f"DNA sequence: {dna_sequence}")
                                
                                if dna_sequence:
                                    # Validate the DNA sequence
                                    is_valid, similarity, best_segment = validate_dna_sequence(
                                        dna_sequence, clean_aa, verbose
                                    )
                                    
                                    # Update best DNA sequence if this one is better
                                    if similarity > best_similarity_score:
                                        best_similarity_score = similarity
                                        best_dna_sequence = best_segment
                                        
                                        # Check if we've found a valid DNA sequence
                                        if similarity >= valid_dna_threshold:
                                            valid_dna_found = True
                                            if verbose:
                                                print(f"Found valid DNA sequence with similarity {similarity:.2f}")
                                            break  # Break out of the gene_results loop
                            
                            # If we still don't have a valid DNA sequence and we have a UniProt ID, try that
                            if not valid_dna_found and gene_info['uniprot_id']:
                                dna_sequence = get_dna_sequence_from_uniprot(gene_info['uniprot_id'])
                                print(f"DNA sequence: {dna_sequence}")
                                
                                if dna_sequence:
                                    # Validate the DNA sequence
                                    is_valid, similarity, best_segment = validate_dna_sequence(
                                        dna_sequence, clean_aa, verbose
                                    )
                                    
                                    # Update best DNA sequence if this one is better
                                    if similarity > best_similarity_score:
                                        best_similarity_score = similarity
                                        best_dna_sequence = best_segment
                                        
                                        # Check if we've found a valid DNA sequence
                                        if similarity >= valid_dna_threshold:
                                            valid_dna_found = True
                                            if verbose:
                                                print(f"Found valid DNA sequence with similarity {similarity:.2f}")
                                            break  # Break out of the gene_results loop
                        
                        # If we've found a valid DNA sequence, break out of the gene_results loop
                        if valid_dna_found:
                            break
    
    # If we haven't found a valid match yet, try BLAST search
    if not found_valid_match or not valid_dna_found:
        if verbose:
            print("Falling back to BLAST search")
        
        blast_results = search_by_blast(clean_aa)
        print(f"BLAST results: {blast_results}")
        if blast_results:
            for blast_result in blast_results:
                if verbose:
                    print(f"Found match via BLAST: {blast_result['uniprot_id']}")
                
                # Update annotation with BLAST info
                annotation.uniprot_id = blast_result['uniprot_id']
                annotation.gene_id = blast_result['gene_id']
                annotation.valid = True
                
                # Try to get DNA sequence if we don't have a valid one yet
                if not valid_dna_found:
                    if 'dna_sequence' in blast_result:
                        dna_sequence = blast_result['dna_sequence']
                    else:
                        gene_info = get_gene_sequence_from_entrez(blast_result['gene_id'])
                        if gene_info:
                            dna_sequence = gene_info['dna_sequence']
                        else:
                            dna_sequence = get_dna_sequence_by_gene_symbol(blast_result['gene_id'])
                        if not dna_sequence:
                            dna_sequence = get_dna_sequence_from_uniprot(blast_result['uniprot_id'])

                    # Validate the DNA sequence
                    if dna_sequence:
                        is_valid, similarity, best_segment = validate_dna_sequence(
                            dna_sequence, clean_aa, verbose
                        )
                        # Update best DNA sequence if this one is better
                        if similarity > best_similarity_score:
                            best_similarity_score = similarity
                            best_dna_sequence = best_segment
                            
                            # Check if we've found a valid DNA sequence
                            if similarity >= valid_dna_threshold:
                                valid_dna_found = True
                                if verbose:
                                    print(f"Found valid DNA sequence with similarity {similarity:.2f}")
                                break  # Break out of the blast_results loop
                
                # If we've found a valid DNA sequence, break out of the blast_results loop
                if valid_dna_found:
                    break
    
    # Set the best DNA sequence we found (if any)
    if best_dna_sequence:
        annotation.dna_sequence = best_dna_sequence
        
        if verbose:
            if valid_dna_found:
                print(f"Using validated DNA sequence with similarity score {best_similarity_score:.2f}")
            else:
                print(f"Using best available DNA sequence with similarity score {best_similarity_score:.2f} (below validation threshold)")
    
    return annotation

