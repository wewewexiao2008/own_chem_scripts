# 新代码
from pathlib import Path

import chemfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chemfp import ob2fps
from chemfp import search
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

sns.set()


def chunk_files(file_list, chunk_size):
    """
    This function splits a list of files into chunks of a specified size.
    """
    for i in range(0, len(file_list), chunk_size):
        yield file_list[i: i + chunk_size] if i + chunk_size <= len(
            file_list
        ) else file_list[i:]


def similarity_matrix(arena):
    n = len(arena)
    ## Compute the full similarity matrix.
    # The implementation computes the upper-triangle then copies
    # the upper-triangle into lower-triangle. It does not include
    # terms for the diagonal.
    results = search.threshold_tanimoto_search_symmetric(arena, threshold=0.0)

    # Extract the results as a SciPy compressed sparse row matrix
    csr = results.to_csr()
    # Convert it to a NumPy array
    similarities = csr.toarray()
    # Fill in the diagonal
    np.fill_diagonal(similarities, 1)

    # Return the distance matrix using the similarity matrix
    return similarities


def distance_matrix(arena):
    return 1.0 - similarity_matrix(arena)


def convert_pdbqt_to_fps(
        input_dir: Path,
        fps_dir: Path,
        chunk_size: int,
        nbits: int,
        fps_type: str,
        use_bitarray: bool,
) -> np.ndarray:
    """
    This function converts pdbqt files to fps files and returns a numpy array of fingerprints.

    Args:
        input_dir: A Path object representing the input directory.
        fps_dir: A Path object representing the output directory for fps files.
        chunk_size: An integer representing the number of files to process at a time.
        nbits: An integer representing the number of bits to use for the fingerprints.
        fps_type: A string representing the type of fingerprint to use.
        use_bitarray: A boolean indicating whether to use bitarray or numpy array for the fingerprints.

    Returns:
        A numpy array of fingerprints.
    """
    file_list = [f for f in input_dir.iterdir() if f.suffix == ".pdbqt"]

    # Calculate the number of files in the directory
    num_files = len(file_list)
    # Calculate the number of chunks

    df_all = pd.DataFrame(columns=["id", "fps", "chunk"])

    fps_dir.mkdir(parents=True, exist_ok=True)  # 创建fps目录

    for i, chunk in tqdm(enumerate(chunk_files(file_list, chunk_size))):
        # 将pdbqt文件转换为fps文件
        ob2fps(
            [str(input_dir / f.name) for f in chunk],
            str(fps_dir / f"chunk_{i}.fps"),
            type=fps_type,
            nBits=nbits,
            progress=False,
        )
        # 从fps文件中读取指纹数据
        arena = chemfp.load_fingerprints(str(fps_dir / f"chunk_{i}.fps"))
        df = pd.DataFrame(
            columns=["id", "fps", "chunk"]
        )  # 创建一个空的DataFrame，用于存储当前chunk的id和fps
        df["id"] = arena.ids
        df["fps"] = (
            list(arena.to_numpy_bitarray())
            if use_bitarray
            else list(arena.to_numpy_array())
        )
        df["chunk"] = i
        df_all = pd.concat([df_all, df])  # 将当前chunk的id和fps添加到总的DataFrame中

    return df_all


def plot_arena(
        arena, fps_df: pd.DataFrame, output_dir: str, use_method: str = "tsne"
) -> None:
    """
    Plot PCA and t-SNE graphs and save them to an svg file.

    Args:
        data_path (Path): Path to the csv file or .
        use_pca (int): Whether to use PCA or t-SNE for dimensionality reduction.
        use_fps (int): Whether to use fingerprints or ro5 for clustering.
    """

    X = arena.to_numpy_array()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # if use_method == "pca" or use_method == 'all':
    # 使用PCA降维
    pca_model = PCA(n_components=2, random_state=42)
    pca_result = pca_model.fit_transform(X)
    fps_df["PC_1"] = pca_result.T[0]
    fps_df["PC_2"] = pca_result.T[1]
    sns.scatterplot(
        x="PC_1",
        y="PC_2",
        hue="cid",
        s=15,
        data=fps_df,
        alpha=0.8,
        edgecolors="none",
        ax=axs[0],
    )
    axs[0].set_title("PCA")

    # if use_method == "tsne":
    # 使用t-SNE降维
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1500)
    tsne_result = tsne_model.fit_transform(X)
    fps_df["tSNE_1"] = tsne_result.T[0]
    fps_df["tSNE_2"] = tsne_result.T[1]
    sns.scatterplot(
        x="tSNE_1",
        y="tSNE_2",
        s=15,
        data=fps_df,
        alpha=0.7,
        hue="cid",
        edgecolors="none",
        ax=axs[1],
    )
    axs[1].set_title("t-SNE")

    # 将图形输出到svg文件
    plt.savefig(output_dir / "pca-tsne.svg", format="svg")
