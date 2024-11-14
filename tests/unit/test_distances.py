# Copyright 2024 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest

from mostlyai.qa.common import TGT_COLUMN_PREFIX
from mostlyai.qa.distances import (
    calculate_distances,
    plot_store_distances,
)
from mostlyai.qa.sampling import calculate_embeddings

N_COMMON = 20
N_RARE = 100


@pytest.fixture()
def distances_dcr(cols_5):
    trn, hol, syn = cols_5
    return trn.add_prefix(TGT_COLUMN_PREFIX), hol.add_prefix(TGT_COLUMN_PREFIX), syn.add_prefix(TGT_COLUMN_PREFIX)


@pytest.fixture()
def too_many_cols():
    categories = ["A", "B", "C", "D", "E", "F", "G"]
    col = categories * 50
    df = pd.DataFrame({f"{TGT_COLUMN_PREFIX}cat_{i}": col for i in range(1100)})
    return df, df, df


@pytest.fixture()
def cat_with_rare_and_none():
    common = [f"common_{i}" for i in range(N_COMMON)] + [None]
    rare = [f"rare_{i}" for i in range(N_RARE)]
    col = common * 10 + rare
    ser = pd.Series(col)
    return ser, ser, ser


def test_calculate_distances():
    syn = pd.DataFrame([{"cat": "a", "n_int": 0, "n_float": 1.0}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    trn = pd.DataFrame([{"cat": "a", "n_int": 0, "n_float": 0.0}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    hol = pd.DataFrame([{"cat": "a", "n_int": 0, "n_float": 1.0}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    syn_embeds, trn_embeds, hol_embeds = calculate_embeddings(syn), calculate_embeddings(trn), calculate_embeddings(hol)
    dcr_trn, dcr_hol = calculate_distances(syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds)
    assert len(dcr_trn) == trn.shape[0]
    assert len(dcr_hol) == hol.shape[0]
    assert dcr_trn.min() > 0
    assert dcr_hol.max() == 0

    # test specifically that near matches do not report a distance of 0 due to rounding
    trn = pd.DataFrame([{"cat": "a", "n_float": 0.0001}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    hol = pd.DataFrame([{"cat": "a", "n_float": 0.0001}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    syn = pd.DataFrame([{"cat": "a", "n_float": 0.0002}] * 10).add_prefix(TGT_COLUMN_PREFIX)
    syn_embeds, trn_embeds, hol_embeds = calculate_embeddings(syn), calculate_embeddings(trn), calculate_embeddings(hol)
    dcr_trn, dcr_hol = calculate_distances(syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds)
    assert dcr_hol.min() > 0


def test_plot_store_dcr(distances_dcr, workspace):
    trn, hol, syn = distances_dcr
    trn_embeds, hol_embeds, syn_embeds = calculate_embeddings(trn), calculate_embeddings(hol), calculate_embeddings(syn)
    dcr_trn, dcr_hol = calculate_distances(syn_embeds=syn_embeds, trn_embeds=trn_embeds, hol_embeds=hol_embeds)
    plot_store_distances(dcr_trn, dcr_hol, workspace)
    output_dir = workspace.workspace_dir / "figures"
    assert len(list(output_dir.glob("*.html"))) == 1
