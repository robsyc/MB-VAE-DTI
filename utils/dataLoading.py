import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.multi_pred import DTI
from typing import Literal
import h5torch
from Bio import Entrez, SeqIO

# sns style
sns.set_style("whitegrid")

# Data structure
# bound: bool (central unit)
# in_Davis: bool (unstructured)
# in_BindingDB: bool (unstructured)
# in_Metz: bool (unstructured)
# in_KIBA: bool (unstructured)

# pKd: float (central feature)
# pKi: float (central feature)
# KIBA_score: float (central feature)

# Target_ID: str (unstructured)
# Targe
# Target_AA: str (unstructured)
# Target_DNA: str (unstructured)

# Drug_ID: str (unstructured)
# Drug_SMILES: str (unstructured)

def format_DTI_cols(df):
    df.rename(columns = {'Drug': 'Drug_SMILES'}, inplace = True)
    df.rename(columns = {'Target': 'Target_AA'}, inplace = True)
    return df.drop(columns = ['Drug_ID', 'Target_ID'])

def plot_distribution(data, value_col, threshold, title):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=value_col, bins=12)
    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.title(f'Distribution of values in dataset {title}')
    plt.xlabel(f'binding affinity in {value_col}')
    plt.ylabel('Frequency')
    plt.show()
    
def print_statistics(df):
    stats = {
        'Unique Drugs': len(df['Drug_SMILES'].unique()),
        'Unique Targets': len(df['Target_AA'].unique()),
        'Total Interactions': len(df),
        'Binding Interactions': df['bound'].sum(),
        'Non-binding Interactions': (~df['bound']).sum()
    }
    print("\nDataset Statistics:")
    print(pd.DataFrame([stats]).T.to_string())

def load_Davis(verbose=False):
    data = DTI(name='DAVIS')
    data.convert_to_log()
    df = data.get_data()
    df.rename(columns={'Y': 'pKd'}, inplace=True)
    df["in_Davis"] = True
    df["bound"] = df["pKd"] >= 7.0
    df = format_DTI_cols(df)

    if verbose:
        plot_distribution(df, 'pKd', 7.0, 'Davis')
        print_statistics(df)
        
    return df

def load_BindingDB_Kd(verbose=False):
    data = DTI(name='BindingDB_Kd')
    data.convert_to_log()
    data.harmonize_affinities(mode='mean')
    df = data.get_data()
    df.rename(columns={'Y': 'pKd'}, inplace=True)
    df["in_BindingDB"] = True
    df["bound"] = df["pKd"] >= 7.0
    df = format_DTI_cols(df)
    
    if verbose:
        plot_distribution(df, 'pKd', 7.0, 'BindingDB_Kd')
        print_statistics(df)
        
    return df

def load_BindingDB_Ki(verbose=False):
    data = DTI(name = 'BindingDB_Ki')
    data.convert_to_log()
    data.harmonize_affinities(mode = 'mean')
    df = data.get_data()
    df.rename(columns = {'Y': 'pKi'}, inplace = True)
    df["in_BindingDB"] = True
    df["bound"] = df["pKi"] >= 7.6
    df = format_DTI_cols(df)

    if verbose:
        plot_distribution(df, 'pKi', 7.6, 'BindingDB_Ki')
        print_statistics(df)
        
    return df

def load_Metz(verbose=False):
    try:
        data = pd.read_csv('data/Metz.csv', usecols=['SMILES', 'ProteinSequence', 'Ki'])
    except FileNotFoundError:
        print('To run the Metz dataset, you need to download it from Kaggle and place it in the data folder.')
        print("https://www.kaggle.com/datasets/blk1804/metz-drug-binding-dataset")
        raise FileNotFoundError
        
    data["in_Metz"] = True
    data.rename(columns={'Ki': 'pKi'}, inplace=True)
    data["bound"] = data["pKi"] >= 7.6
    data.rename(columns={'ProteinSequence': 'Target_AA', 'SMILES': 'Drug_SMILES'}, inplace=True)
    
    if verbose:
        plot_distribution(data, 'pKi', 7.6, 'Metz')
        print_statistics(data)
        
    return data

def load_KIBA(verbose=False):
    data = DTI(name = 'KIBA')
    df = data.get_data()
    df.rename(columns = {'Y': 'KIBA_score'}, inplace = True)
    df["in_KIBA"] = True
    df["bound"] = df["KIBA_score"] >= 12.1
    df = format_DTI_cols(df)
    
    if verbose:
        plot_distribution(df, 'KIBA_score', 12.1, 'KIBA')
        print_statistics(df)
        
    return df