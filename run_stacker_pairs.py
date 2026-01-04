import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.ndimage as ndi
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))  # libs/ under code/

from libs.models import STaCkerSemiSupervised
from libs.utils import simple_efm_kwargs_generator
from libs import alignment as align_mod



def require_env(name):
    v = os.environ.get(name, "")
    if str(v).strip() == "":
        raise ValueError("Missing required env var: %s" % name)
    return str(v).strip()

def env_int(name, default):
    v = os.environ.get(name, "")
    v = str(v).strip()
    if v == "":
        return int(default)
    return int(v)

def parse_pairs(pairs_str):
    # format:
    #   meta_ref,meta_mov,expr_ref,expr_mov; meta_ref2,meta_mov2,expr_ref2,expr_mov2
    items = []
    for chunk in str(pairs_str).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "STACKER_PAIRS format error.\n"
                "Expected: meta_ref,meta_mov,expr_ref,expr_mov; meta_ref2,meta_mov2,expr_ref2,expr_mov2\n"
                "Got: %s" % chunk
            )
        items.append(tuple(parts))
    if not items:
        raise ValueError("STACKER_PAIRS parsed empty.")
    return items



def read_csv_any(path):
    # pandas can read .csv and .csv.gz directly
    return pd.read_csv(path)

def load_meta_expr(meta_path, expr_path, xcol, ycol):
    meta = read_csv_any(meta_path)

    if "cell_id" not in meta.columns:
        meta.rename(columns={meta.columns[0]: "cell_id"}, inplace=True)
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta = meta.set_index("cell_id")

    meta[xcol] = pd.to_numeric(meta[xcol], errors="coerce")
    meta[ycol] = pd.to_numeric(meta[ycol], errors="coerce")
    meta = meta.dropna(subset=[xcol, ycol])

    # expr read
    expr = read_csv_any(expr_path)

 
    expr2 = pd.read_csv(expr_path, index_col=0)
    expr2.index = expr2.index.astype(str)
    expr2.columns = expr2.columns.astype(str)

    overlap_cols = len(set(expr2.columns) & set(meta.index))
    overlap_idx = len(set(expr2.index) & set(meta.index))

    if overlap_cols == 0 and overlap_idx == 0:
        exprT = expr2.T
        exprT.index = exprT.index.astype(str)
        exprT.columns = exprT.columns.astype(str)
        overlap_cols2 = len(set(exprT.columns) & set(meta.index))
        overlap_idx2 = len(set(exprT.index) & set(meta.index))
        if max(overlap_cols2, overlap_idx2) == 0:
            raise ValueError(
                "No overlapping cell IDs between meta and expr.\n"
                "meta n=%d ; expr shape=%s ; overlap(meta, expr.columns)=%d ; overlap(meta, expr.index)=%d"
                % (len(meta), str(expr2.shape), overlap_cols, overlap_idx)
            )
        expr2 = exprT
        overlap_cols = overlap_cols2
        overlap_idx = overlap_idx2


    if overlap_cols >= overlap_idx:
        expr_gc = expr2
    else:
        expr_gc = expr2.T

    expr_gc.index = expr_gc.index.astype(str)
    expr_gc.columns = expr_gc.columns.astype(str)

    common = meta.index.intersection(expr_gc.columns)
    if len(common) == 0:
        raise ValueError("No overlapping cell IDs after orientation fix.")

    meta = meta.loc[common]
    expr_gc = expr_gc[common]

    coords = meta[[xcol, ycol]].to_numpy(dtype=np.float32)
    return meta, expr_gc, coords


def to_adata(meta, expr_gc, coords):
    X = expr_gc.T.astype(np.float32)
    adata = ad.AnnData(sp.csr_matrix(X.values))
    adata.obs_names = meta.index.astype(str)
    adata.var_names = expr_gc.index.astype(str)
    adata.obsm["spatial"] = coords.astype(np.float32)
    return adata


# Density image (for STaCker)
def estimate_sigma(coords, k=5):
    coords = np.asarray(coords, dtype=np.float32)
    k = min(int(k), len(coords))
    if k <= 1:
        return 2.0
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    dists, _ = nn.kneighbors(coords)
    return float(np.median(dists[:, 1]) / 2.0 + 1e-6)

def attach_density_image(adata, out_size=512, sigma=3.0, key="hires", library_id="library"):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    ok = np.isfinite(coords).all(axis=1)
    coords = coords[ok]
    if coords.shape[0] == 0:
        raise RuntimeError("No valid coords to build density image.")

    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    coords01 = (coords - mn) / span
    pix = np.round(coords01 * (out_size - 1)).astype(int)

    img = np.zeros((out_size, out_size), dtype=np.float32)
    xs = np.clip(pix[:, 0], 0, out_size - 1)
    ys = np.clip(pix[:, 1], 0, out_size - 1)
    np.add.at(img, (ys, xs), 1.0)

    img = ndi.gaussian_filter(img, sigma=float(sigma))
    if img.max() <= 0:
        img[0, 0] = 1.0
    img = img / img.max()

    rgb = np.stack([img] * 3, axis=-1).astype(np.float32)

    adata.uns["spatial"] = {
        library_id: {
            "images": {key: rgb},
            "scalefactors": {
                "tissue_%s_scalef" % key: 1.0,
                "spot_diameter_fullres": float(2.0 * sigma),
            },
        }
    }


# run one pair
def run_one_pair(cfg, meta_ref, meta_mov, expr_ref, expr_mov):
    tag = "%s__VS__%s" % (Path(meta_ref).stem, Path(meta_mov).stem)
    print("[STaCker] Running pair:", tag)

    data_dir = Path(cfg["DATA_DIR"])

    ref_meta, ref_expr_gc, ref_xy = load_meta_expr(
        str(data_dir / meta_ref), str(data_dir / expr_ref), cfg["XCOL"], cfg["YCOL"]
    )
    mov_meta, mov_expr_gc, mov_xy = load_meta_expr(
        str(data_dir / meta_mov), str(data_dir / expr_mov), cfg["XCOL"], cfg["YCOL"]
    )

    ref_adata = to_adata(ref_meta, ref_expr_gc, ref_xy)
    mov_adata = to_adata(mov_meta, mov_expr_gc, mov_xy)

    sigma = estimate_sigma(ref_adata.obsm["spatial"], k=cfg["SIGMA_K"])
    attach_density_image(ref_adata, out_size=cfg["DENSITY_SIZE"], sigma=sigma, key="hires", library_id="library")
    attach_density_image(mov_adata, out_size=cfg["DENSITY_SIZE"], sigma=sigma, key="hires", library_id="library")
    print("[STaCker] density_size=%d sigma=%.3f" % (cfg["DENSITY_SIZE"], sigma))

    # build model
    model = STaCkerSemiSupervised(
        img_shape=(256, 256, 3),
        lbl_shape=(256, 256, 3),
        auxiliary_outputs=["def_output", "inv_def_output"],
        efm_kwargs=simple_efm_kwargs_generator(
            dict_inputs=True,
            dict_outputs=True,
            output_names_list=["moved_img", "def_output", "inv_def_output", "moved_lbl"],
        ),
    )
    model.load_weights(cfg["WEIGHTS"])
    model = model.references.base_model

    aligned = align_mod.stacker_register(
        slices=[ref_adata, mov_adata],
        alignment_mode="templated",
        ref_index=0,
        spatial_strategy="points",
        spatial_target="hires",
        max_reg_dim=cfg["MAX_REG_DIM"],
        mode="dense",
        model=model,
        verbose=False,
    )

    ref_aligned, mov_aligned = aligned

    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ref = out_dir / ("%s_ref.csv" % tag)
    out_mov = out_dir / ("%s_mov_aligned.csv" % tag)

    pd.DataFrame(
        {
            "cell_id": ref_aligned.obs_names.astype(str),
            "x": ref_aligned.obsm["spatial"][:, 0],
            "y": ref_aligned.obsm["spatial"][:, 1],
        }
    ).to_csv(out_ref, index=False)

    pd.DataFrame(
        {
            "cell_id": mov_aligned.obs_names.astype(str),
            "x_aligned": mov_aligned.obsm["spatial"][:, 0],
            "y_aligned": mov_aligned.obsm["spatial"][:, 1],
        }
    ).to_csv(out_mov, index=False)

    print("[STaCker] Saved:", out_ref)
    print("[STaCker] Saved:", out_mov)


def main():
    cfg = {}
    cfg["DATA_DIR"] = require_env("STACKER_DATA_DIR")
    cfg["OUT_DIR"] = require_env("STACKER_OUT_DIR")
    cfg["PAIRS"] = require_env("STACKER_PAIRS")
    cfg["WEIGHTS"] = require_env("STACKER_WEIGHTS")

    cfg["XCOL"] = os.environ.get("STACKER_XCOL", "center_x").strip()
    cfg["YCOL"] = os.environ.get("STACKER_YCOL", "center_y").strip()

    cfg["MAX_REG_DIM"] = env_int("STACKER_MAX_REG_DIM", 512)
    cfg["DENSITY_SIZE"] = env_int("STACKER_DENSITY_SIZE", 512)
    cfg["SIGMA_K"] = env_int("STACKER_SIGMA_K", 5)

    pairs = parse_pairs(cfg["PAIRS"])
    for meta_ref, meta_mov, expr_ref, expr_mov in pairs:
        run_one_pair(cfg, meta_ref, meta_mov, expr_ref, expr_mov)


if __name__ == "__main__":
    main()
