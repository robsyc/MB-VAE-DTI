import os
import time
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

# Define paths relative to the project structure
# Assuming this script is run from the project root
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# --- Constants ---
# NCBI Taxonomy IDs
SPECIES_TAX_IDS = {
    "human": 9606,
    "chimp": 9598,
    "mouse": 10090,
    # "yeast": 559292,  # S. cerevisiae S288c
    # "arabidopsis": 3702, # A. thaliana
    # "ecoli": 83333 # E. coli K-12 MG1655
}

# Filtering criteria
MAX_AA_LENGTH = 1280

# NCBI Entrez settings
Entrez.email = "robbe.claeys@ugent.be" # IMPORTANT: Replace with your actual email
FETCH_BATCH_SIZE = 200 # Number of records to fetch per Entrez request
MAX_RECORDS_PER_SPECIES = 300000 # Target records per species before filtering
REQUEST_DELAY = 0.4 # Seconds to wait between Entrez requests (to stay under 3/sec)

# Output file
OUTPUT_FILE = PROCESSED_DIR / "pretrain_targets.csv"

# --- Function Stubs ---

def fetch_sequences_for_species(tax_id: int, species_name: str) -> List[Dict[str, str]]:
    """
    Fetches protein (aa) and coding DNA (dna) sequences for a given species
    from NCBI Nucleotide database, filtering by protein length.
    """
    if Entrez.email == "your.email@example.com":
        print("🛑 ERROR: Please set your email address for Entrez before running.")
        return []
        
    print(f"\n🧬 Fetching sequences for {species_name} (TaxID: {tax_id})...")
    sequences = []
    retrieved_count = 0
    processed_count = 0
    
    try:
        # 1. Search for relevant nucleotide record IDs
        print(f"  Searching NCBI Nucleotide for records... (Max: {MAX_RECORDS_PER_SPECIES})")
        search_term = f"txid{tax_id}[Organism:exp] AND (biomol mrna[PROP] OR cds[Feature key]) NOT (" \
                      f"environmental sample[filter] OR metagenome[filter] OR unverified[filter])"
        
        print(f"  Using search term: {search_term}")
        handle = Entrez.esearch(
            db="nuccore", 
            term=search_term,
            idtype="acc",
            retmax=MAX_RECORDS_PER_SPECIES
        )
        search_results = Entrez.read(handle)
        handle.close()
        ids = search_results["IdList"]
        print(f"  Found {len(ids)} potential record IDs.")

        if not ids:
            print(f"  No records found for {species_name} with the current criteria.")
            return []

        # 2. Fetch records in batches
        for start in tqdm(range(0, len(ids), FETCH_BATCH_SIZE), desc=f"Fetching {species_name} records"):
            end = min(len(ids), start + FETCH_BATCH_SIZE)
            batch_ids = ids[start:end]
            
            try:
                # 3. Fetch GenBank records
                time.sleep(REQUEST_DELAY) # Respect NCBI rate limits
                handle = Entrez.efetch(
                    db="nuccore", 
                    id=batch_ids, 
                    rettype="gb", 
                    retmode="text"
                )
                
                # 4. Parse records
                for record in SeqIO.parse(handle, "genbank"):
                    processed_count += 1
                    found_cds_in_record = False # Flag for debugging
                    # 5. Extract CDS features
                    for feature in record.features:
                        if feature.type == "CDS":
                            found_cds_in_record = True # Found at least one CDS
                            # Ensure required qualifiers are present
                            # Removed "coded_by" check as feature.extract uses location
                            if "translation" not in feature.qualifiers:
                                # print(f"  Skipping CDS in {record.id}: Missing 'translation' qualifier.")
                                continue 
                                
                            try:
                                # Extract AA sequence from qualifier
                                aa_seq_str = feature.qualifiers["translation"][0]
                                # print(f"  Found CDS in {record.id}. AA length: {len(aa_seq_str)}. Seq starts: {aa_seq_str[:10]}...") # Debug print
                                
                                # 6. Filter by AA length
                                if len(aa_seq_str) > MAX_AA_LENGTH:
                                    # print(f"    -> Rejected: AA length {len(aa_seq_str)} > {MAX_AA_LENGTH}") # Debug print
                                    continue
                                    
                                # Check for ambiguous AAs or internal stop codons
                                if 'X' in aa_seq_str or '*' in aa_seq_str[:-1]: # Allow stop codon only at the very end
                                    # print(f"    -> Rejected: Contains 'X' or internal '*'.") # Debug print
                                    continue

                                # Extract DNA sequence using the feature location
                                dna_seq = feature.extract(record.seq)
                                dna_seq_str = str(dna_seq)
                                # print(f"    Extracted DNA length: {len(dna_seq_str)}. Starts: {dna_seq_str[:10]}...") # Debug print

                                # Basic validation of DNA sequence
                                # Calculate expected length based on AA seq excluding any terminal stop codon
                                aa_len_no_stop = len(aa_seq_str.rstrip('*'))
                                expected_dna_len_no_stop = aa_len_no_stop * 3
                                
                                # Check if DNA length is either exactly matching the coding region
                                # OR matching the coding region + 3 bases for the stop codon.
                                if not (len(dna_seq_str) == expected_dna_len_no_stop or 
                                        len(dna_seq_str) == expected_dna_len_no_stop + 3):
                                     # print(f"    -> Rejected: DNA length {len(dna_seq_str)} doesn't match expected {expected_dna_len_no_stop} or {expected_dna_len_no_stop + 3} based on AA length {aa_len_no_stop}.") # Debug print
                                     continue # Mismatch length
                                         
                                if not all(c in 'ATCGN' for c in dna_seq_str.upper()):
                                    # print(f"    -> Rejected: Contains non-standard DNA bases.") # Debug print
                                    continue # Contains non-standard DNA bases
                                    
                                # 7. Store valid pair (Remove terminal stop from AA if present)
                                final_aa_seq = aa_seq_str.rstrip('*')
                                sequences.append({"aa": final_aa_seq, "dna": dna_seq_str})
                                retrieved_count += 1
                                # print(f"    -> Accepted: Stored pair.") # Debug print
                                
                            except Exception as e:
                                # Log errors during feature extraction/processing
                                print(f"  ⚠️ Error processing feature in record {record.id} (feature location: {feature.location}): {e}")
                                # Keep passing for now to avoid stopping the whole batch
                                pass 
                    
                    #if not found_cds_in_record:
                        # print(f"  Record {record.id}: No CDS features found.") # Debug print
                                
                handle.close()

            except Exception as e:
                print(f"\n⚠️ Error fetching or parsing batch {start // FETCH_BATCH_SIZE + 1}: {e}")
                print(f"  Problematic IDs might be: {batch_ids}")
                print(f"  Skipping this batch.")
                time.sleep(5) # Longer delay after error
                continue # Continue to the next batch
                
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred during the overall fetch process for {species_name}: {e}")
        # Potentially re-raise or handle more gracefully depending on needs

    print(f"  Processed {processed_count} records.")
    print(f"✅ Finished fetching for {species_name}. Retrieved {retrieved_count} valid sequences.")
    return sequences

def main():
    """
    Main function to fetch sequences for all defined species,
    combine them, remove duplicates, and save to a CSV file.
    """
    all_sequences = []
    print("Starting sequence fetching process...")

    for name, tax_id in SPECIES_TAX_IDS.items():
        species_sequences = fetch_sequences_for_species(tax_id, name)
        all_sequences.extend(species_sequences)
        print(f"Collected {len(species_sequences)} sequences for {name}.")
        print(f"Total sequences so far: {len(all_sequences)}")

    if not all_sequences:
        print("No sequences were fetched. Exiting.")
        return

    print(f"Total sequences collected before deduplication: {len(all_sequences)}")

    # Convert to DataFrame and deduplicate
    df = pd.DataFrame(all_sequences)
    df.drop_duplicates(subset=['aa', 'dna'], inplace=True)
    print(f"Total unique sequences after deduplication: {len(df)}")

    # Save the final dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} unique protein/CDS pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    # IMPORTANT: Make sure to set your email address for Entrez
    if Entrez.email == "your.email@example.com":
        print("🛑 Please replace 'your.email@example.com' with your actual email address in the script.")
    else:
        main() 