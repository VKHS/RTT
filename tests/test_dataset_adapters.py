"""Raw-format and portable-format checks for every manuscript dataset family."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtt.datasets import DATASETS, DatasetError, load_dataset
from rtt.datasets.good_arxiv import bundle_from_good


def _image(path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(50, 100, 150)).save(path)


def _portable(path, name):
    task = "graph" if name == "ogbg_molpcba" else "image" if name == "imagenet_c" else "tabular" if name == "acs_income" else "node"
    values = dict(name=name, task=task, num_classes=2,
                  y=np.array([0, 1, 0, 1, 0, 1]), train_idx=[0, 1], valid_idx=[2, 3], test_idx=[4, 5])
    if task in {"node", "tabular"}:
        values["x"] = np.arange(24, dtype=np.float32).reshape(6, 4)
    if task == "node":
        values["edge_index"] = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    if task == "graph":
        labels = np.zeros((6, 128), dtype=np.float32)
        labels[:, 0] = 1; labels[:, 3] = np.nan
        values.update(num_classes=128, y=labels,
                      x=np.zeros((12, 9), dtype=np.int64),
                      edge_index=np.stack([np.arange(12), np.arange(12) ^ 1]),
                      edge_attr=np.zeros((12, 3), dtype=np.int64),
                      graph_ptr=np.arange(0, 13, 2), edge_ptr=np.arange(0, 13, 2))
    if task == "image":
        paths=[]
        for i in range(6):
            file=path.parent/f"image_{i}.jpg"; _image(file); paths.append(file.name)
        values["image_paths"] = np.asarray(paths)
    np.savez(path, **values)
    return values


@pytest.mark.parametrize("name", DATASETS)
def test_all_registered_datasets_load_explicit_portable_schema(tmp_path, name):
    path = tmp_path / (name + ".npz")
    _portable(path, name)
    result = load_dataset(name.replace("_", "-"), path)
    assert result.name == name and result.n == 6
    assert len(result.splits["train"]) == 2 and result.year is None
    if name == "ogbg_molpcba":
        assert result.multilabel and result.num_classes == 128
        assert np.isnan(result.y[:, 3]).all()
        assert result.graphs[1].edge_index.max() == 1
        assert result.graphs[0].x.dtype == np.int64
    assert len(result.fingerprint()) == 64


def test_portable_rejects_wrong_dataset_overlap_and_unknown_scored_label(tmp_path):
    path=tmp_path/"data.npz"; values=_portable(path, "ogbn_arxiv")
    with pytest.raises(DatasetError, match="cannot be used"):
        load_dataset("ogbn_products", path)
    values["test_idx"]=[0,5]; np.savez(path, **values)
    with pytest.raises(DatasetError, match="overlap"):
        load_dataset("ogbn_arxiv", path)
    values["test_idx"]=[4,5]; values["y"][2]=-1; np.savez(path,**values)
    with pytest.raises(DatasetError, match="unknown labels"):
        load_dataset("ogbn_arxiv", path)


@pytest.mark.parametrize("name", ["ogbn_arxiv", "ogbn_products"])
def test_official_ogb_node_api_preserves_split_and_year_semantics(tmp_path, monkeypatch, name):
    (tmp_path/name).mkdir()
    class Dataset:
        num_classes=2
        def __init__(self, name, root): self.name=name
        def __getitem__(self,index):
            graph={"node_feat":np.arange(24).reshape(6,4),"edge_index":np.array([[0,1],[1,2]])}
            if self.name=="ogbn-arxiv": graph["node_year"]=np.array([2017,2017,2018,2018,2019,2020])[:,None]
            return graph,np.array([0,1,0,1,0,1])[:,None]
        def get_idx_split(self):return {"train":np.array([0,1]),"valid":np.array([2,3]),"test":np.array([4,5])}
    module=types.ModuleType("ogb.nodeproppred"); module.NodePropPredDataset=Dataset
    monkeypatch.setitem(sys.modules,"ogb",types.ModuleType("ogb"))
    monkeypatch.setitem(sys.modules,"ogb.nodeproppred",module)
    data=load_dataset(name,tmp_path)
    np.testing.assert_allclose(data.x[data.splits["train"]].mean(0),0,atol=1e-6)
    assert (data.year is None)==(name=="ogbn_products")
    np.testing.assert_array_equal(data.splits["test"],[4,5])


def test_ogb_molecule_api_preserves_categorical_bonds_and_nan_tasks(tmp_path, monkeypatch):
    from rtt.models import ATOM_CARDINALITIES, BOND_CARDINALITIES
    (tmp_path/"ogbg_molpcba").mkdir()
    class Dataset:
        def __init__(self,**kwargs):
            self.labels=np.zeros((3,128),dtype=np.float32);self.labels[:,2]=np.nan
        def __len__(self):return 3
        def __getitem__(self,index):
            atoms=np.zeros((2,9),dtype=np.int64)
            atoms[:,1]=4  # Official OGB chirality vocabulary includes the fifth "misc" category.
            return {"node_feat":atoms,"edge_feat":np.ones((2,3),dtype=np.int64),
                    "edge_index":np.array([[0,1],[1,0]])},self.labels[index]
        def get_idx_split(self):return {"train":np.array([0]),"valid":np.array([1]),"test":np.array([2])}
    module=types.ModuleType("ogb.graphproppred");module.GraphPropPredDataset=Dataset
    monkeypatch.setitem(sys.modules,"ogb",types.ModuleType("ogb"));monkeypatch.setitem(sys.modules,"ogb.graphproppred",module)
    data=load_dataset("ogbg_molpcba",tmp_path)
    assert data.multilabel and np.isnan(data.y[:,2]).all()
    assert data.graphs[0].edge_attr.dtype==np.int64 and data.graphs[0].x.shape==(2,9)
    assert np.all(data.graphs[0].x[:,1]==4)
    assert tuple(data.metadata["atom_cardinalities"])==ATOM_CARDINALITIES==(119,5,12,12,10,6,6,2,2)
    assert tuple(data.metadata["bond_cardinalities"])==BOND_CARDINALITIES==(5,6,2)


@pytest.mark.parametrize("domain,shift", [("degree","covariate"),("time","concept")])
def test_good_uses_released_masks_instead_of_ogb_year_masks(domain,shift):
    raw={"x":np.arange(24).reshape(6,4),"y":np.array([0,1,0,1,0,1]),
         "edge_index":np.array([[0,1],[1,2]]),"train_mask":np.array([1,0,1,0,0,0],dtype=bool),
         "val_mask":np.array([0,1,0,1,0,0],dtype=bool),"test_mask":np.array([0,0,0,0,1,1],dtype=bool)}
    data=bundle_from_good(raw,domain,shift)
    np.testing.assert_array_equal(data.splits["train"],[0,2])
    assert data.metadata["shift"]==shift and data.year is None
    raw.pop("val_mask")
    with pytest.raises(DatasetError,match="official mask"):
        bundle_from_good(raw,domain,shift)


def _write_twitch(root,name,n):
    folder=root/name;folder.mkdir()
    (folder/f"musae_{name}_target.csv").write_text("new_id,mature\n"+"".join(f"{i},{bool(i%2)}\n" for i in range(n)))
    (folder/f"musae_{name}_features.json").write_text(json.dumps({str(i):[i%4] for i in range(n)}))
    (folder/f"musae_{name}_edges.csv").write_text("from,to\n"+"".join(f"{i},{i+1}\n" for i in range(n-1)))


def test_twitch_raw_language_graphs_keep_feature_ids_and_component_offsets(tmp_path):
    _write_twitch(tmp_path,"DE",6);_write_twitch(tmp_path,"ENGB",4)
    data=load_dataset("twitch",tmp_path,{"source_domain":"DE","target_domains":["ENGB"]})
    assert data.x.shape==(10,4) and data.splits["test"].min()==6
    assert all((u<6)==(v<6) for u,v in data.edge_index.T)
    np.testing.assert_array_equal(data.x[0],data.x[6])
    with pytest.raises(DatasetError,match="Specify source"):
        load_dataset("twitch",tmp_path)


def test_facebook_mat_excludes_gender_features_and_keeps_unknown_nodes(tmp_path):
    info=np.array([[1,1,20,2007],[1,2,21,2008],[1,0,20,2007],[2,1,22,2009],[2,2,21,2008],[1,1,20,2010]])
    for name in ("Source","Target"):
        savemat(tmp_path/(name+".mat"),{"A":sparse.eye(6),"local_info":info})
    opts={"source_domain":"Source","target_domains":["Target"]}
    data=load_dataset("facebook100",tmp_path,opts)
    assert data.y[2]==-1 and 2 not in data.splits["train"] and 8 not in data.splits["test"]
    changed=info.copy();changed[:,1]=np.where(changed[:,1]>0,3-changed[:,1],0)
    savemat(tmp_path/"Source.mat",{"A":sparse.eye(6),"local_info":changed})
    again=load_dataset("facebook100",tmp_path,opts)
    np.testing.assert_array_equal(data.x,again.x)
    assert not np.array_equal(data.y,again.y)


def test_elliptic_csv_keeps_unknown_context_and_temporal_split(tmp_path):
    rows=np.array([[100,1,1,10],[101,1,2,20],[200,2,3,30],[201,2,4,40],[300,3,5,50],[301,3,6,60]])
    np.savetxt(tmp_path/"elliptic_txs_features.csv",rows,fmt="%d",delimiter=",")
    (tmp_path/"elliptic_txs_classes.csv").write_text("txId,class\n100,1\n101,2\n200,2\n201,unknown\n300,1\n301,2\n")
    (tmp_path/"elliptic_txs_edgelist.csv").write_text("txId1,txId2\n100,101\n200,201\n300,301\n")
    data=load_dataset("elliptic",tmp_path,{"train_end":1,"valid_end":2})
    np.testing.assert_array_equal(data.splits["test"],[4,5])
    assert data.y[3]==-1 and data.y[0]==1 and data.year is None and data.time[4]==3
    assert data.x.shape[1]==2


def test_acs_calls_official_problem_and_fits_scaling_only_in_source(tmp_path,monkeypatch):
    calls=[]
    class Source:
        def __init__(self,**kwargs):calls.append(kwargs)
        def get_data(self,states,download):return states[0]
    class Income:
        @staticmethod
        def df_to_numpy(state):
            x=np.arange(60,dtype=float).reshape(6,10)+(0 if state=="CA" else 1000)
            return x,np.array([0,1,0,1,0,1]),np.zeros(6)
    module=types.ModuleType("folktables");module.ACSDataSource=Source;module.ACSIncome=Income
    monkeypatch.setitem(sys.modules,"folktables",module)
    data=load_dataset("acs_income",tmp_path,{"target_states":["NY","TX"]})
    assert data.n==18 and len(data.splits["test"])==12
    np.testing.assert_allclose(data.x[data.splits["train"]].mean(0),0,atol=1e-6)
    assert data.x[data.splits["test"]].mean()>10
    assert calls[0]["root_dir"]==str(tmp_path)


def test_imagenet_official_folders_use_fixed_classes_and_lazy_paths(tmp_path):
    clean=tmp_path/"clean";corrupt=tmp_path/"corrupt"
    for cls in ("n0001","n0002"):
        for i in range(3):_image(clean/cls/f"train_{cls}_{i}.JPEG")
        for corruption in ("snow","fog"):
            for i in range(2):_image(corrupt/corruption/"3"/cls/f"test_{cls}_{i}.JPEG")
    data=load_dataset("imagenet_c",corrupt,{"fit_root":str(clean),"num_classes":2,"corruptions":["snow","fog"]})
    assert data.task=="image" and data.x is None and data.n==14
    assert len(data.splits["test"])==8 and data.metadata["severity"]==3
    assert data.metadata["class_to_idx"]=={"n0001":0,"n0002":1}


def test_image_manifest_rejects_same_underlying_image_across_splits(tmp_path):
    for name in ("clean.jpg","snow.jpg","valid.jpg"):_image(tmp_path/name)
    path=tmp_path/"images.csv"
    path.write_text("path,label,split,image_id\nclean.jpg,0,train,image1\nvalid.jpg,1,valid,image2\nsnow.jpg,0,test,image1\n")
    with pytest.raises(DatasetError,match="crosses splits"):
        load_dataset("imagenet_c",tmp_path,{"manifest":str(path),"num_classes":2})


def test_portable_image_ids_cannot_bypass_identity_split_check(tmp_path):
    path=tmp_path/"images.npz";values=_portable(path,"imagenet_c")
    values["image_ids"]=np.array(["shared","b","c","d","shared","f"])
    np.savez(path,**values)
    with pytest.raises(DatasetError,match="crosses splits"):
        load_dataset("imagenet_c",path)


def test_good_pickle_requires_explicit_trust_before_loading(tmp_path):
    path=tmp_path/"untrusted.pt";path.write_bytes(b"not a checkpoint")
    with pytest.raises(DatasetError,match="trust_data_pickle"):
        load_dataset("good_arxiv_degree",tmp_path,{"data_pt":str(path)})
