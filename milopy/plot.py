import scanpy as sc
import pandas as pd
import numpy as np
import anndata
from typing import Union, Optional, Sequence, Any, Mapping, List, Tuple
from anndata import AnnData

import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_nhood_graph(
    adata: AnnData,
    alpha: float = 0.1,
    min_logFC: float = 0,
    min_size: int = 10,
    plot_edges: bool = False,
    title: str = "DA log-Fold Change",
    **kwargs
):
    '''
    Visualize DA results on abstracted graph (wrapper around sc.pl.embedding)

    Params:
    -------
    - adata: AnnData object
    - alpha: significance threshold
    - min_logFC: minimum absolute log-Fold Change to show results (default: 0, show all significant neighbourhoods)
    - min_size: minimum size of nodes in visualization (default: 10)
    - plot_edges: boolean indicating if edges for neighbourhood overlaps whould be plotted (default: False)
    - title: plot title (default: 'DA log-Fold Change')
    - **kwargs: other arguments to pass to scanpy.pl.embedding
    '''
    nhood_adata = adata.uns["nhood_adata"].copy()

    if "Nhood_size" not in nhood_adata.obs.columns:
        raise KeyError(
            'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                please run milopy.utils.build_nhood_graph(adata)'
        )
    
   # find all logFC columns
    logfc_cols = [c for c in nhood_adata.obs.columns if c.startswith("logFC")]

    # helper to prepare a sub‐AnnData for a given contrast
    def _prepare(sub_adata, fc_col):
        obs = sub_adata.obs.copy()
        obs["graph_color"] = obs[fc_col]
        # mask non‐significant by FDR
        obs.loc[obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        # mask small effects
        obs["abs_logFC"] = obs[fc_col].abs()
        obs.loc[obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan
        # ensure masked points go to bottom
        obs.loc[obs["graph_color"].isna(), "abs_logFC"] = np.nan
        # reorder cells
        ordered = obs.sort_values("abs_logFC", na_position="first").index
        sub = sub_adata[ordered].copy()
        sub.obs = obs.loc[ordered]
        return sub

    # multiple‐contrast grid
    if len(logfc_cols) > 1:
        n = len(logfc_cols)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

        for ax, fc_col in zip(axes, logfc_cols):
            sub = _prepare(nhood_adata, fc_col)
            vmax = max(sub.obs["graph_color"].max(), -sub.obs["graph_color"].min())
            vmin = -vmax

            sc.pl.embedding(
                sub,
                basis="X_milo_graph",
                color="graph_color",
                cmap="RdBu_r",
                size=sub.obs["Nhood_size"] * min_size,
                edges=plot_edges,
                neighbors_key="nhood",
                sort_order=False,
                frameon=False,
                vmax=vmax,
                vmin=vmin,
                ax=ax,
                show=False,
                **kwargs
            )
            ax.set_title(fc_col)

        # hide any unused axes
        for ax in axes[len(logfc_cols) :]:
            ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fig

    # single‐contrast fallback
    fc_col = logfc_cols[0] if logfc_cols else "logFC"
    sub = _prepare(nhood_adata, fc_col)
    vmax = max(sub.obs["graph_color"].max(), -sub.obs["graph_color"].min())
    vmin = -vmax

    sc.pl.embedding(
        sub,
        basis="X_milo_graph",
        color="graph_color",
        cmap="RdBu_r",
        size=sub.obs["Nhood_size"] * min_size,
        edges=plot_edges,
        neighbors_key="nhood",
        sort_order=False,
        frameon=False,
        vmax=vmax,
        vmin=vmin,
        title=title,
        **kwargs
    )


def plot_nhood(adata, ix, basis="X_umap"):
    '''
    Visualize cells in a neighbourhood
    '''
    adata.obs["Nhood"] = adata.obsm["nhoods"][:, ix].toarray().ravel()
    sc.pl.embedding(adata, basis, color="Nhood",
                    size=30, title="Nhood" + str(ix))

# Plot nhood beeswarm


def plot_DA_beeswarm(
    adata: AnnData,
    anno_col: str = 'nhood_annotation',
    alpha: float = 0.1,
    subset_nhoods: List = None
):
    '''
    Plot beeswarm plot of logFC against nhood labels

    Params:
    -------
    - adata: AnnData object
    - anno_col: column in adata.uns['nhood_adata'].obs to use as annotation
    - alpha: significance threshold
    - subset_nhoods: list of nhoods to plot (default: None, plot all nhoods)
    '''
    try:
        nhood_adata = adata.uns["nhood_adata"].copy()
    except KeyError:
        raise KeyError(
            'Cannot find "nhood_adata" slot in adata.uns -- please run milopy.make_nhoods_adata(adata)'
        )

    if subset_nhoods is not None:
        nhood_adata = nhood_adata[subset_nhoods]

    try:
        nhood_adata.obs[anno_col]
    except KeyError:
        raise KeyError(
            'Cannot find {a} in adata.uns["nhood_adata"].obs -- \
                please run milopy.utils.annotate_nhoods(adata, anno_col) first'.format(a=anno_col)
        )

    try:
        nhood_adata.obs["logFC"]
    except KeyError:
        raise KeyError(
            'Cannot find `logFC` in adata.uns["nhood_adata"].obs -- \
                please run milopy.core.DA_nhoods(adata) first'
        )

    sorted_annos = nhood_adata.obs[[anno_col, "logFC"]].\
        groupby(anno_col).\
        median().\
        sort_values("logFC", ascending=True).index

    anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR"]].copy()
    anno_df['is_signif'] = anno_df['SpatialFDR'] < alpha
    # anno_df['across_organs'] = ["Significant across organs" if x else "" for x in (keep_nhoods & signif_nhoods)]
    anno_df = anno_df[anno_df[anno_col] != "nan"]

    try:
        anno_palette = _get_palette_adata(
            adata, nhood_adata.uns['annotation_obs'])
        sns.violinplot(data=anno_df, y=anno_col, x="logFC", order=sorted_annos,
                       size=190, inner=None, orient="h",
                       palette=anno_palette,
                       linewidth=0,
                       scale="width")
    except:
        sns.violinplot(data=anno_df, y=anno_col, x="logFC", order=sorted_annos,
                       size=190, inner=None, orient="h",
                       linewidth=0,
                       scale="width")
    sns.stripplot(data=anno_df, y=anno_col, x="logFC", order=sorted_annos, size=2,
                  hue='is_signif', palette=['grey', 'black'],
                  orient="h", alpha=0.5)
    plt.legend(loc='upper left', title=f'< {int(alpha*100)}% SpatialFDR',
               bbox_to_anchor=(1, 1), frameon=False)
    plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")


def _get_palette_adata(adata, obs_col):
    return(dict(zip(adata.obs[obs_col].cat.categories, adata.uns[f'{obs_col}_colors'])))


### Plot boxplot of cell numbers for QC ###
def plot_nhood_counts_by_cond(
        adata: AnnData,
        test_var: str,
        subset_nhoods: List = None,
        log_counts: bool = False):
    '''
    Plot boxplot of cell numbers vs condition of interest

    Params:
    ------
    - adata: anndata object storing neighbourhood information in adata.uns
    - test_var: string, name of column in adata.obs storing condition of interest (y-axis for boxplot)
    - subset_nhoods: list of obs_names for neighbourhoods to include in plot (default: None, plot all nhoods)
    - log_counts: boolean, whether to plot log1p of cell counts (default: False)
    '''
    try:
        nhood_adata = adata.uns["nhood_adata"].copy()
    except KeyError:
        raise KeyError(
            'Cannot find "nhood_adata" slot in adata.uns -- please run milopy.make_nhoods_adata(adata)'
        )

    if subset_nhoods is None:
        subset_nhoods = nhood_adata.obs_names

    pl_df = pd.DataFrame(nhood_adata[subset_nhoods].X.A, columns=nhood_adata.var_names).melt(
        var_name=nhood_adata.uns['sample_col'], value_name='n_cells')
    pl_df = pd.merge(pl_df, nhood_adata.var)
    pl_df['log_n_cells'] = np.log1p(pl_df['n_cells'])
    if not log_counts:
        sns.boxplot(data=pl_df, x=test_var, y='n_cells', color='lightblue')
        sns.stripplot(data=pl_df, x=test_var, y='n_cells', color='black', s=3)
        plt.ylabel('# cells')
    else:
        sns.boxplot(data=pl_df, x=test_var, y='log_n_cells', color='lightblue')
        sns.stripplot(data=pl_df, x=test_var,
                      y='log_n_cells', color='black', s=3)
        plt.ylabel('log(# cells + 1)')

    plt.xticks(rotation=90)
    plt.xlabel(test_var)
