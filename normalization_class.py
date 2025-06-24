# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:59:45 2025

@author: User
"""

import pandas as pd
import tensorflow as tf
import json, os
from typing import List, Union, Dict, Tuple, Any

class DataSummary:
    """
    DataSummary

    A utility to manage and query statistical summaries for numeric features/records.

    Supports inputs:
      - pandas DataFrame
      - dict of sequences → DataFrame
      - (dict, index_list) tuple → DataFrame
      - JSON file path (outer keys→rows, inner dicts→columns)
      - dict of dicts (JSON-like)

    Combines all into one DataFrame, stores stats in `self.statistics` (shape [R, F]),
    preserves per-record `shape` metadata (`self.shapes`), and provides case-insensitive
    lookups (`lookup`, `by_index`, `get_key`, `keys`, `all_stats`, `get_shape`).

    Includes `normalize()` for pure TF graph-mode normalization, delegating index-map
    construction to `create_statistics_index_full` for robust handling of scalar,
    vector, and 2×K mapping indices.
    """

    def __init__(
        self,
        data_list: List[Union[pd.DataFrame, Dict[str, Any], Tuple[Dict, List[str]], str]],
        dtype: tf.DType = tf.float32
    ):
        self.dtype = dtype
        self.shapes: Dict[str, List[int]] = {}

        dfs: List[pd.DataFrame] = []
        for item in data_list:
            if isinstance(item, pd.DataFrame):
                df = item.copy()
            elif isinstance(item, str) and os.path.isfile(item) and item.lower().endswith('.json'):
                with open(item, 'r') as f:
                    data_dict = json.load(f)
                df = pd.DataFrame.from_dict(data_dict, orient='index')
            elif isinstance(item, dict) and all(isinstance(v, dict) for v in item.values()):
                df = pd.DataFrame.from_dict(item, orient='index')
            elif isinstance(item, tuple) and isinstance(item[0], dict) and isinstance(item[1], list):
                data_dict, index_list = item
                df = pd.DataFrame(data_dict, index=index_list)
            elif isinstance(item, dict):
                df = pd.DataFrame(item)
            else:
                raise TypeError(
                    "Each element must be DataFrame, JSON path, nested dict, dict, or (dict,index_list) tuple."
                )

            if 'shape' in df.columns:
                for idx_key, shp in df['shape'].items():
                    self.shapes[str(idx_key).lower()] = shp
                df = df.drop(columns=['shape'])

            dfs.append(df)

        combined = pd.concat(dfs, axis=0, ignore_index=False) if dfs else pd.DataFrame()
        self.statistics = tf.convert_to_tensor(combined.values, dtype=self.dtype)

        self.x_keys = [col.lower() for col in combined.columns]
        self.y_keys = [str(idx).lower() for idx in combined.index]
        self._x_lookup = {k: i for i, k in enumerate(self.x_keys)}
        self._y_lookup = {k: i for i, k in enumerate(self.y_keys)}
        self._reverse_lookup = {
            **{i: k for k, i in self._x_lookup.items()},
            **{i: k for k, i in self._y_lookup.items()}
        }

    def lookup(self, key: str) -> tf.Tensor:
        lk = key.lower()
        if lk in self._x_lookup:
            return self.statistics[:, self._x_lookup[lk]]
        if lk in self._y_lookup:
            return self.statistics[self._y_lookup[lk], :]
        raise KeyError(f"Key '{key}' not found.")

    def by_index(self, idx: int) -> tf.Tensor:
        if not (0 <= idx < tf.shape(self.statistics)[0]):
            raise IndexError(f"Index {idx} out of range.")
        return self.statistics[idx, :]

    def get_key(self, idx: int) -> Union[str, None]:
        return self._reverse_lookup.get(idx)

    def keys(self) -> Dict[str, List[str]]:
        return {'x': self.x_keys, 'y': self.y_keys}

    def all_stats(self) -> tf.Tensor:
        return self.statistics

    def get_shape(self, key: str) -> List[int]:
        lk = key.lower()
        if lk in self.shapes:
            return self.shapes[lk]
        raise KeyError(f"Shape for key '{key}' not found.")

    def create_statistics_index_full(
        self,
        nonorm_input: tf.Tensor,
        statistics_index: Union[int, tf.Tensor],
        normalization_dimension: int
    ) -> tf.Tensor:
        """
        Builds a full-map tensor of the same shape as `nonorm_input`, mapping each element
        along `normalization_dimension` to a statistics row index.

        - Scalar index → filled tensor.
        - 1D vector of length D → reshaped and broadcast.
        - 2×K mapping tensor → scatter into length-D then reshape and broadcast.
        """
        nonorm_input = tf.convert_to_tensor(nonorm_input)
        in_shape = tf.shape(nonorm_input)
        idx = tf.convert_to_tensor(statistics_index, dtype=tf.int32)
        ndims = tf.rank(nonorm_input)
        norm_dim = tf.where(normalization_dimension < 0,
                            normalization_dimension + ndims,
                            normalization_dimension)
        D = in_shape[norm_dim]

        def _build_scalar_mapping(scalar_idx):
            positions = tf.cast(
                tf.linspace(0.0, tf.cast(D - 1, tf.float32), tf.cast(D, tf.int32)),
                tf.int32
            )
            rows = tf.repeat(scalar_idx, D)
            return tf.stack([positions, rows], axis=0)

        def _build_full_from_mapping(map_idx):
            positions = map_idx[0]
            rows = map_idx[1]
            row_map = tf.fill([D], -1)
            indices = tf.expand_dims(positions, 1)
            row_map = tf.tensor_scatter_nd_update(row_map, indices, rows)
            # broadcast shape
            broadcast_shape = tf.ones([ndims], dtype=tf.int32)
            broadcast_shape = tf.tensor_scatter_nd_update(
                broadcast_shape, [[norm_dim]], [D]
            )
            row_map_rs = tf.reshape(row_map, broadcast_shape)
            return tf.broadcast_to(row_map_rs, in_shape)

        rank_idx = tf.rank(idx)
        # If scalar, convert to 2×D mapping
        map_idx = tf.cond(
            tf.equal(rank_idx, 0),
            lambda: _build_scalar_mapping(idx),
            lambda: idx
        )
        # Ensure map_idx is 2×K
        map_shape = tf.shape(map_idx)
        tf.debugging.assert_equal(map_shape[0], 2,
                                  message="Mapping must have shape [2, K]")
        # Always build full from mapping
        return _build_full_from_mapping(map_idx)

    @tf.function(jit_compile=False)
    def normalize(
        self,
        nonorm_input: tf.Tensor,
        norm_config: Dict[str, Any] = {'Norm_Limits': (0.0, 1.0), 'Input_Normalization': 'lnk-linear-scaling'},
        statistics_index: Union[int, tf.Tensor] = tf.constant(-1),
        compute: bool = False,
        normalization_dimension: int = -1,
        dtype: tf.DType = tf.float32
    ) -> tf.Tensor:
        """
        Apply per-slice normalization to `nonorm_input` using stored statistics.

        Delegates index-map creation to `create_statistics_index_full` for robust handling
        of scalar, vector, and 2×K mappings. Normalization schemes: lnk-linear, linear,
        and z-score; only applies where index>=0.
        """
        if not compute:
            return nonorm_input

        nonorm_input = tf.convert_to_tensor(nonorm_input, dtype=dtype)
        # Build full index map
        stats_idx_full = self.create_statistics_index_full(
            nonorm_input, statistics_index, normalization_dimension
        )
        # Gather per-element stats [min,max,mean,std,count]
        ts = tf.gather(self.statistics, stats_idx_full)
        norm_min, norm_max = norm_config['Norm_Limits']

        # Normalization functions
        def _lnk():
            no_log = (((nonorm_input - ts[...,0])/(ts[...,1]-ts[...,0]))*(norm_max-norm_min))+norm_min
            log    = ((tf.math.log(nonorm_input/ts[...,0])/
                       tf.math.log(ts[...,1]/ts[...,0]))*(norm_max-norm_min))+norm_min
            cond   = tf.logical_and(
                tf.not_equal(stats_idx_full, 5),
                tf.not_equal(stats_idx_full, 6)
            )
            return tf.where(cond, no_log, log)

        def _lin():
            return (((nonorm_input - ts[...,0])/(ts[...,1]-ts[...,0]))*(norm_max-norm_min))+norm_min

        def _z():
            return (nonorm_input - ts[...,2]) / ts[...,3]

        base = tf.where(
            tf.equal(norm_config['Input_Normalization'], 'lnk-linear-scaling'),
            _lnk(), _lin()
        )
        normed = tf.where(
            tf.equal(norm_config['Input_Normalization'], 'z-score'),
            _z(), base
        )

        # Apply mask: only index>=0
        out = tf.where(tf.greater_equal(stats_idx_full, 0), normed, nonorm_input)
        # Clean NaN/Inf
        return tf.where(
            tf.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)),
            tf.zeros_like(out), out
        )

# Example usage
if __name__ == '__main__':
    summary = DataSummary([{
        'feat1': {'min':0,'max':10,'mean':5,'std':2,'count':100},
        'feat2': {'min':0,'max':20,'mean':10,'std':4,'count':200}
    }])

    inp = tf.constant([[1.,2.,3.,4.],[5.,6.,7.,8.]])  # Shape [2,4]
    stats_idx_scalar = tf.constant(1, dtype=tf.int32)
    stats_idx_map = tf.constant([[0,2],[0,1]], dtype=tf.int32)

    cfg = {'Norm_Limits':(0.,1.), 'Input_Normalization':'linear-scaling'}
    out_scalar = summary.normalize(inp, cfg, stats_idx_scalar, True)
    out_map    = summary.normalize(inp, cfg, stats_idx_map,    True)
    #print("Scalar-normalized:\n", out_scalar.numpy())
    print("Map-normalized:\n", out_map.numpy())
