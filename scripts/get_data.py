import os
from utils.dataProcessing import load_df, explore_df
from utils.dataProcessing import generate_h5torch
from utils.preEmbedding import add_embeddings

os.chdir('/home/robbec/thesis/MB-VAE-DTI/')

df_davis = load_df(
    name="DAVIS",
    use_filters=True,
    to_log=True,
    seed=42
)

for df, name in zip([df_davis], ["DAVIS"]):
    print(f"\n{name} dataset")
    print(f"Number of drugs\t\t: {df['Drug'].nunique():,}")
    print(f"Number of targets\t: {df['Target'].nunique():,}")
    print(f"Number of interactions\t: {len(df):,}")

generate_h5torch(df_davis, "DAVIS")

add_embeddings("DAVIS")