import shutil
# 新代码
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors

from rdkit2pdbqt import MolFromPDBQTBlock


def _read_pdbqt(file_path):
    with open(file_path) as f:
        try:
            res = MolFromPDBQTBlock(f.read()), file_path.stem.split("_")[1]
        except:
            res = None, None
    return res


def _cal_mol_props(smi, verbose=False):
    try:
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None, None, None, None, None, None, None, None
        mw = round(Descriptors.MolWt(m), 1)
        logp = round(Descriptors.MolLogP(m), 2)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(m)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(m)
        psa = round(Descriptors.TPSA(m), 1)
        rob = rdMolDescriptors.CalcNumRotatableBonds(m)
        qed = round(QED.qed(m), 2)
        chiral_center = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        if verbose:
            print("Mw ", mw)
            print("Logp ", logp)
            print("HBD ", hbd)
            print("HBA ", hba)
            print("TPSA ", psa)
            print("RotB ", rob)
            print("QED ", qed)
            print("chiral_center ", chiral_center)
        return mw, logp, hbd, hba, psa, rob, qed, chiral_center
    except Exception as e:
        print(e)
        return None, None, None, None, None, None, None, None


def read_pdbqts(input_dir: Path, visualize=True):
    """
    Process pdbqt files in a directory and return a pandas dataframe with relevant information.

    Args:
        input_dir (Path): Path object representing the directory containing pdbqt files.

    Returns:
        pd.DataFrame: A pandas dataframe with columns for pdbqt file name, molecule object, molecule ID, SMILES string, and visual representation of the molecule.
    """
    file_list = [f for f in input_dir.iterdir() if f.suffix == ".pdbqt"]

    my_df = pd.DataFrame(file_list, columns=["path"])

    my_df["mol"], my_df["nid"] = zip(
        *my_df["path"].parallel_apply(lambda f: _read_pdbqt(f))
    )
    my_df = my_df.dropna(subset=["nid"])

    my_df["smiles"] = my_df["mol"].parallel_apply(
        lambda x: Chem.MolToSmiles(x) if x is not None else None
    )

    if visualize:
        my_df["mol_vis"] = my_df["smiles"].parallel_apply(
            lambda x: Chem.MolFromSmiles(x)
        )
        # @todo output image of molecules
        PandasTools.FrameToGridImage(
            my_df,
            column="mol_vis",
            molsPerRow=9,
            subImgSize=(100, 100),
            legendsCol="nid",
        )

    return my_df


def filter_molecules(
        my_df: pd.DataFrame, thresh: dict, output_dir: Path
) -> pd.DataFrame:
    """
    Filter molecules based on certain properties and save the filtered dataframe to a csv file.

    Args:
        my_df (pd.DataFrame): A pandas dataframe with columns for pdbqt file name, molecule object, molecule ID, SMILES string, and visual representation of the molecule.
        thresh (dict): A dictionary for threshold.
        output_dir (Path): Path object representing the directory where the filtered dataframe and pdbqt files will be saved.

    Returns:
        pd.DataFrame: A pandas dataframe with the filtered molecules.
    """
    my_df[["Mw", "logP", "HBD", "HBA", "TPSA", "RotB", "QED", "chiral"]] = (
        my_df["smiles"].parallel_apply(_cal_mol_props).apply(pd.Series)
    )

    filter_df = my_df.copy()

    for i in thresh.keys():
        filter_df = filter_df[
            (filter_df[i] > thresh[i][0]) & (filter_df[i] < thresh[i][1])
            ]
    print("一共有{}个分子".format(len(my_df)))
    print("筛选后剩余{}个分子".format(len(filter_df)))

    output_dir.mkdir(parents=True, exist_ok=True)
    filter_df["path"].parallel_apply(lambda x: shutil.copyfile(x, output_dir / x.name))

    return filter_df


def collect_repr_from_df(
        df: pd.DataFrame,
        pdbqt_dir: Path,
        cluster_dir: Path,
        fig_dir: Path,
        copy_raw: bool,
) -> pd.DataFrame:
    repr_ls = []
    for cid in sorted(df.index.get_level_values("cid").unique()):
        c_df = get_cluster_df(df, (cid))
        # sort by nid from filename and get the first one

        repr_ls.append(c_df.sort_values("_nid").iloc[0])
        # to image
        img = visual_cluster(c_df.reset_index())
        of = fig_dir / f"{cid}_mols.svg"
        of.write_text(img)

        if copy_raw:
            # copy file to separate clus dir
            cid_dir = cluster_dir / str(cid)
            cid_dir.mkdir(parents=True, exist_ok=True)
            for pdbqt_file in df.loc[df.index.get_level_values("cid") == cid, "path"]:
                src_path = pdbqt_dir / pdbqt_file.name
                dst_path = cid_dir / pdbqt_file.name
                shutil.copy(src_path, dst_path)

    repr_df = pd.DataFrame(repr_ls)
    repr_dir = cluster_dir / "_repr"
    repr_dir.mkdir(parents=True, exist_ok=True)
    for i, pdbqt_file in enumerate(repr_df["path"]):
        src_path = pdbqt_dir / pdbqt_file.name
        dst_path = repr_dir / str(f"c{i}_" + pdbqt_file.name)
        shutil.copy(src_path, dst_path)
    return repr_df


def get_cluster_df(df_join, cluster_id, include_all=True):
    """_summary_

    Args:
        df_join (df): joined df
        cluster_id (tuple): cluster id tuple, e.g. (1,2)

    Returns:
        df: cluster df
    """
    # Mw	logP	HBD	HBA	TPSA	RotB	QED	chiral	tpsa
    if include_all:
        return df_join.loc[(slice(None), cluster_id), :]
    else:
        return df_join.loc[
            (slice(None), cluster_id), ["path", "mol_vis", "_nid", "_name"]
        ]


def visual_cluster(df):
    return PandasTools.FrameToGridImage(
        df,
        column="mol_vis",
        molsPerRow=9,
        subImgSize=(100, 100),
        # maxMols=1000,
        legendsCol="_name",
        useSVG=True,
        returnPNG=True,
    )
