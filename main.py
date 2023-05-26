import datetime
import logging
import math
import time
from argparse import ArgumentParser
from contextlib import contextmanager

import chemfp
import matplotlib
from pandarallel import pandarallel
from sklearn.cluster import AgglomerativeClustering

from fps import *
from mol_utils import *

# from visualize_clus import *

matplotlib.use('Agg')

NBITS = 1024
FPS_TYPE = "ECFP2"  # in 0,2,4,6,8,10
CHUNK_SIZE = 1000000
USE_BITARRAY = False

N_CLUSTERS = None
DISTANCE_THRESH = 0.5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@contextmanager
def timing(description: str):
    start = time.time()
    yield
    ellapsed_time = time.time() - start
    logger.info(f"{description}: {ellapsed_time:.2f} s")


if __name__ == "__main__":
    pandarallel.initialize(progress_bar=False)

    parser = ArgumentParser(description="Process some directories.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="/public/home/zhangyangroup/public/Zangsk/Virtualflow/H2R/20221115/Chemdiv_samll1_20230330/pp/top900/first",
        help="input directory",
    )  # 修改参数名，添加默认值
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./output/test" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        help="output directory",
    )  # 保持不变，添加默认值
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=0,
        help="whether to filter molecules",
    )
    parser.add_argument("-n", "--n_clusters", type=int, default=N_CLUSTERS, help="n_clusters")
    parser.add_argument("-d", "--distance", type=int, default=DISTANCE_THRESH, help="distance_thresh")
    args = parser.parse_args()
    # 初始化并行库

    n_clusters = args.n_clusters
    distance_thresh = args.distance
    if n_clusters is not None:
        distance_thresh = None
    if n_clusters is not None and distance_thresh is not None:
        raise ValueError("n_clusters and distance_thresh cannot be set at the same time")

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)  # 修改变量名，简化路径处理

    work_dir = output_dir
    pdbqt_dir = input_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    meta_df = read_pdbqts(input_dir)
    if args.threshold:
        with timing("filter molecules"):
            filter_dir = output_dir / "filtered"
            work_dir = filter_dir
            pdbqt_dir = work_dir / "pdbqt"
            pdbqt_dir.mkdir(parents=True, exist_ok=True)

            # 设置阈值
            thresh = {
                "Mw": [0, 450],
                "logP": [2, 5],
                "HBD": [0, 3],
                "HBA": [0, 10],
                "RotB": [0, 9],
                "QED": [0, 10],
                "chiral": [0, 10],
                "TPSA": [0, 90],
            }
            meta_df = filter_molecules(meta_df, thresh, pdbqt_dir)

    # path, nid, mol, smiles, (Mw, logP, HBD, HBA, RotB, QED, chiral, tpsa)
    meta_df.to_csv(work_dir / "metadata.csv", index=False)
    logger.info("metadata.csv saved")

    fps_dir = work_dir / "fps"

    with timing("convert pdbqt to fps"):
        # id, fps, chunk, (cid)
        fps_df = convert_pdbqt_to_fps(
            pdbqt_dir, fps_dir, CHUNK_SIZE, NBITS, FPS_TYPE, USE_BITARRAY
        )

    file_list = [f for f in pdbqt_dir.iterdir() if f.suffix == ".pdbqt"]
    num_files = len(file_list)
    num_chunks = math.ceil(num_files / CHUNK_SIZE)
    labels_all = []
    # print("start clustering")
    with timing("clustering"):
        for i in range(num_chunks):
            arena = chemfp.load_fingerprints(str(fps_dir / f"chunk_{i}.fps"))
            dis_mat = distance_matrix(arena)
            sim_mat = 1 - dis_mat
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="complete",
                distance_threshold=distance_thresh,
            ).fit(dis_mat)
            labels_all.extend(clustering.labels_)
            # visualize
        fps_df["cid"] = labels_all

    # 一定要这个顺序，因为df2的index顺序=arena.ids
    with timing("join"):
        df1 = meta_df[meta_df["nid"].notnull()].set_index("nid")
        df2 = fps_df.set_index("id")
        df_join = df2.join(df1)
        df_join = df_join.reset_index().set_index(["id", "cid"])

        # _df = df_join.loc[(slice(None)), ["path", "smiles", "chunk", "mol_vis"]]

        df_join["_name"] = df_join.apply(lambda x: x["path"].name, axis=1)
        df_join["_nid"] = df_join.apply(lambda x: int(x["_name"].split("_")[0]), axis=1)
        df_join.reset_index()[
            ["id", "_nid", "cid", "smiles", "chunk", "_name", "path"]
        ].to_csv(work_dir / "clustered_joined.csv")

    cluster_dir = work_dir / "clus"
    if cluster_dir.exists():
        shutil.rmtree(cluster_dir)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    repr_ls = []
    # 将每个cid的pdbqt文件单独复制到一个文件夹，原始目录为work_dir,目标地为cluster_dir/{cid}
    fig_dir = work_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    copy_raw = True
    with timing("copy represenative pdbqt files"):
        repr_df = collect_repr_from_df(
            df_join, pdbqt_dir, cluster_dir, fig_dir, copy_raw
        )

    with timing("plot arena"):
        plot_arena(arena, df_join, fig_dir)

    # visualize with pca
    # with timing("visualize with pca"):
    #     v_df = df_join[['']]
