import importlib
import logging

import click
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdmolfiles as rdm
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.DataStructs import TanimotoSimilarity as tanimoto
from sklearn.metrics.pairwise import euclidean_distances

from bmfm_sm.api.dataset_registry import DatasetRegistry
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy


class Fingerprints:
    def __init__(self, rad=3, nBits=1024):
        self.rad = rad
        self.nBits = nBits

    def get_fingerprint(self, smi, fp="morgan"):
        mol = rdm.MolFromSmiles(smi)
        if mol == None:
            return None
        if fp == "morgan":
            return GetMorganFingerprintAsBitVect(mol, self.rad, nBits=self.nBits)
        if fp == "MACCS":
            return MACCSkeys.GenMACCSKeys(mol)
        if fp == "rdkit":
            return self.rdkit_fpgen.GetFingerprint(mol)
        if fp == "torsion":
            return Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        if fp == "atompairs":
            return Pairs.GetAtomPairFingerprintAsBitVect(mol)
        return None

    def get_tanimoto_distance(self, fp1, fp2):
        return 1 - tanimoto(fp1, fp2)


class Assessment:
    def __init__(
        self,
        num_mol=10000,
        file_path="resources/predictive_assessment_tanimoto_dist.csv",
        smile1_col="smile1",
        smile2_col="smile2",
    ):
        if file_path == "resources/predictive_assessment_tanimoto_dist.csv":
            file_path = importlib.resources.files("bmfm_sm.resources").joinpath(
                "predictive_assessment_tanimoto_dist.csv"
            )

        self.smile1_col = smile1_col
        self.smile2_col = smile2_col
        self.df = pd.read_csv(file_path)
        assert num_mol <= len(self.df)
        self.df = self.df[:num_mol]

    def get_correlation(
        self,
        fp_algo,
        model_path=None,
        finetuned=False,
        ft_dataset=None,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,
        all_modality=False,
    ):
        if finetuned:
            model = SmallMoleculeMultiViewModel.from_finetuned(
                dataset=ft_dataset, model_path=model_path, inference_mode=True
            )
        else:
            model = SmallMoleculeMultiViewModel.from_pretrained(
                fusion_strategy=fusion_strategy,
                model_path=model_path,
                inference_mode=True,
            )

        euclidean_dist = []
        if all_modality:
            euclidean_dist_modality = {"graph": [], "image": [], "text": []}

        for i, row in self.df.iterrows():
            if i % 100 == 0:
                logging.info(f"Num of molecules done: {i}")

            if all_modality:
                all_emb1 = SmallMoleculeMultiViewModel.get_embeddings(
                    row[self.smile1_col],
                    pretrained_model=model,
                    get_separate_embeddings=True,
                )
                all_emb2 = SmallMoleculeMultiViewModel.get_embeddings(
                    row[self.smile2_col],
                    pretrained_model=model,
                    get_separate_embeddings=True,
                )

                euclidean_dist_modality["graph"].append(
                    euclidean_distances(
                        all_emb1["Graph2dModel"], all_emb2["Graph2dModel"]
                    )[0][0]
                )
                euclidean_dist_modality["image"].append(
                    euclidean_distances(all_emb1["ImageModel"], all_emb2["ImageModel"])[
                        0
                    ][0]
                )
                euclidean_dist_modality["text"].append(
                    euclidean_distances(all_emb1["TextModel"], all_emb2["TextModel"])[
                        0
                    ][0]
                )
                euclidean_dist.append(
                    euclidean_distances(all_emb1["aggregator"], all_emb2["aggregator"])[
                        0
                    ][0]
                )

            else:
                omni_emb1 = SmallMoleculeMultiViewModel.get_embeddings(
                    row[self.smile1_col], pretrained_model=model
                )
                omni_emb2 = SmallMoleculeMultiViewModel.get_embeddings(
                    row[self.smile2_col], pretrained_model=model
                )
                euclidean_dist.append(
                    euclidean_distances([omni_emb1], [omni_emb2])[0][0]
                )

        result = {}
        self.df["euclidean_dist_smmv"] = euclidean_dist
        for dist_col_name in fp_algo:
            result[dist_col_name] = np.corrcoef(
                euclidean_dist, self.df[dist_col_name].tolist()
            )[0][1]

        if all_modality:
            for modality in ["graph", "image", "text"]:
                self.df[f"euclidean_dist_{modality}"] = euclidean_dist_modality[
                    modality
                ]
                modality_results = {}
                for dist_col_name in fp_algo:
                    modality_results[dist_col_name] = np.corrcoef(
                        euclidean_dist_modality[modality],
                        self.df[dist_col_name].tolist(),
                    )[0][1]
                logging.info(f"Correlation for {modality} model: {modality_results}")

        return result, self.df


datasets = DatasetRegistry.get_instance().list_datasets()
fusion_strategies = [strategy.name.lower() for strategy in LateFusionStrategy]


@click.command()
@click.option(
    "--file_path",
    type=str,
    default="resources/predictive_assessment_tanimoto_dist.csv",
    help="Path to the input file",
)
@click.option(
    "--num_mol", type=int, default=10000, help="Number of molecules to run on"
)
@click.option(
    "--fp_algo",
    type=click.Choice(["morgan", "maccs", "rdkit", "torsion", "atompairs", "all"]),
    default="all",
    help="Fingerprinting algorithm to use",
)
@click.option(
    "--model_path", type=str, required=False, help="Path to the model checkpoint"
)
@click.option(
    "--finetuned",
    type=bool,
    default=False,
    help="Use finetuned model (over pretrained)",
)
@click.option(
    "--ft_dataset",
    type=click.Choice(datasets),
    required=False,
    help="Which dataset the model is finetuned on",
)
@click.option(
    "--all_modality",
    type=bool,
    default=False,
    help="Calculate assessment for all modalities",
)
@click.option(
    "--fusion_strategy",
    type=click.Choice(fusion_strategies),
    required=False,
    help="Late Fusion strategy used by pretrained model",
    default="attentional",
)
@click.option(
    "--save_file_path",
    type=str,
    required=True,
    help="Path to save the output dataframe",
)
def main(
    file_path,
    num_mol,
    fp_algo,
    model_path,
    finetuned,
    ft_dataset,
    all_modality,
    fusion_strategy,
    save_file_path,
):
    # Validating the arguments
    if fp_algo == "all":
        fp_algo = ["morgan", "maccs", "rdkit", "torsion", "atompairs"]
    else:
        fp_algo = [fp_algo]

    if finetuned and ft_dataset is None:
        raise ValueError("ft_dataset value must be provided for a finetuned model.")
    elif finetuned and ft_dataset is not None:
        ft_dataset = DatasetRegistry.get_instance().get_dataset_info(ft_dataset)

    fusion_strategy = LateFusionStrategy.from_string(fusion_strategy)

    logging.info(
        f"Running Assessment Pipeline with SMMV Model, (Finetuned: {finetuned}, Dataset: {ft_dataset}), Fingerprint Algorithm(s): {fp_algo}, Number of Molecules: {num_mol}, Tanimoto Distance File: {file_path}, Model checkpoint: {model_path}"
    )

    # Create the assessment object and run Assessment
    assess = Assessment(num_mol=num_mol, file_path=file_path)
    results, results_df = assess.get_correlation(
        fp_algo,
        model_path=model_path,
        finetuned=finetuned,
        ft_dataset=ft_dataset,
        fusion_strategy=fusion_strategy,
        all_modality=all_modality,
    )

    logging.info(f"Correlation for SMMV model: {results}")
    logging.info(f"Saving results dataframe to {save_file_path}")
    results_df.to_csv(save_file_path)


if __name__ == "__main__":
    main()
