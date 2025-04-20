import unittest

import PIL
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from torch_geometric.data import Data

import bmfm_sm.core.data_modules.fuse_ops.ops_graphs
import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.ops_images import OpSmilesToImage

FIELD_INPUT = "input"
FIELD_OUTPUT = "output"


class TestOpSmilesToGraph(unittest.TestCase):
    def test_constructor(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        assert isinstance(op, OpBase)

    def test_call_1(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        sample_dict = NDict()
        sample_dict[FIELD_INPUT] = "CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"
        sample_dict = op(sample_dict, key_in=FIELD_INPUT, key_out=FIELD_OUTPUT)
        assert sample_dict[FIELD_INPUT] == "CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"
        assert isinstance(sample_dict[FIELD_OUTPUT], Data)
        # there should be 9 atom (node) features
        assert sample_dict[FIELD_OUTPUT].x.shape[1] == 9
        # there should be 5 bond (edge) features
        assert sample_dict[FIELD_OUTPUT].edge_attr.shape[1] == 5

    def test_call_2(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        sample_dict = NDict()
        sample_dict[FIELD_INPUT] = "COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC"
        sample_dict = op(sample_dict, key_in=FIELD_INPUT, key_out=FIELD_OUTPUT)
        assert sample_dict[FIELD_INPUT] == "COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC"
        assert isinstance(sample_dict[FIELD_OUTPUT], Data)
        # there should be 9 atom (node) features
        assert sample_dict[FIELD_OUTPUT].x.shape[1] == 9
        # there should be 5 bond (edge) features
        assert sample_dict[FIELD_OUTPUT].edge_attr.shape[1] == 5

    # Test a problematic SMILE
    def test_call_3(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        sample_dict = NDict()
        sample_dict[
            FIELD_INPUT
        ] = "[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[Fe+6].[Fe]"
        sample_dict = op(sample_dict, key_in=FIELD_INPUT, key_out=FIELD_OUTPUT)
        assert (
            sample_dict[FIELD_INPUT]
            == "[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[Fe+6].[Fe]"
        )
        assert isinstance(sample_dict[FIELD_OUTPUT], Data)
        # there should be 9 atom (node) features
        assert sample_dict[FIELD_OUTPUT].x.shape[1] == 9
        # there should be 5 bond (edge) features
        assert sample_dict[FIELD_OUTPUT].edge_attr.shape[1] == 5

    def test_op_smiles_to_image(self):
        op = OpSmilesToImage(size=224)
        sample_dict = NDict()
        sample_dict[ns.FIELD_DATA_LIGAND_SMILES] = "cccc1C(=O)NCC1(O)CCOCC1"
        sample_dict = op(sample_dict)
        assert isinstance(sample_dict[ns.FIELD_DATA_LIGAND_IMAGE], PIL.Image.Image)


class TestOpGraphToGraphLaplacian(unittest.TestCase):
    def test_constructor(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpGraphToGraphLaplacian()
        assert isinstance(op, OpBase)

    def test_call_1(self):
        FIELD_SMILES = "smiles"
        FIELD_GRAPH = "graph"

        op1 = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        sample_dict = NDict()
        sample_dict[FIELD_SMILES] = "COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC"
        sample_dict = op1(sample_dict, key_in=FIELD_SMILES, key_out=FIELD_GRAPH)
        assert sample_dict[FIELD_SMILES] == "COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC"
        assert isinstance(sample_dict[FIELD_GRAPH], Data)
        # there should be 9 atom (node) features
        assert sample_dict[FIELD_GRAPH].x.shape[1] == 9
        # there should be 5 bond (edge) features
        assert sample_dict[FIELD_GRAPH].edge_attr.shape[1] == 5

        op2 = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpGraphToGraphLaplacian()
        sample_dict = op2(sample_dict, key_in=FIELD_GRAPH, key_out=FIELD_GRAPH)
        assert sample_dict[FIELD_GRAPH]["lap_eigvec"] is not None
        assert sample_dict[FIELD_GRAPH]["lap_eigvec"].shape[0] == 20
        assert sample_dict[FIELD_GRAPH]["lap_eigval"] is not None
        assert sample_dict[FIELD_GRAPH]["lap_eigval"].shape[0] == 20


class TestOpGraphToRankHomology(unittest.TestCase):
    def test_constructor(self):
        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpGraphToRankHomology()
        assert isinstance(op, OpBase)

    def test_call_1(self):
        FIELD_SMILES = "smiles"
        FIELD_BETTI = "betti"

        bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpSmilesToGraph()
        sample_dict = NDict()
        sample_dict[FIELD_SMILES] = "COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC"

        op = bmfm_sm.core.data_modules.fuse_ops.ops_graphs.OpGraphToRankHomology()
        sample_dict = op(sample_dict, key_in=FIELD_SMILES, key_out=FIELD_BETTI)
        # There should be 20 betti number pairs
        assert len(sample_dict[FIELD_BETTI]) == 20


if __name__ == "__main__":
    unittest.main()
