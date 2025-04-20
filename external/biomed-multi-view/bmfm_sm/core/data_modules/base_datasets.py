from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict
from torch.utils.data import Dataset

from bmfm_sm.core.data_modules.fuse_ops.ops_general import OpReadDataframeBmfm


class FuseFeaturizedDataset(Dataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        # data_dir and stage are typically used by the subclasses to implemented their reading of the data and the next_item() function

        # get_ops_pipeline should be implemented by the subclasses
        self.all_ops_and_kwargs = self.get_ops_pipeline()._ops_and_kwargs

        if isinstance(self.all_ops_and_kwargs[0][0], OpReadDataframeBmfm):
            self.all_ops_and_kwargs.pop(0)

        # Other core arguments
        self.dataset_args = dataset_args
        self.prefix_map = dataset_args.get("prefix_map", None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the raw data first
        item_data = self.next_item(idx)

        # Format it the way that the pipeline expects so it doesn't have to do the read
        if self.prefix_map:
            input_ndict = NDict()
            for name, value in item_data.items():
                if name not in self.prefix_map.keys():
                    input_ndict[name] = value
                else:
                    input_ndict[f"{self.prefix_map[name]}.{name}"] = value
        else:
            input_ndict = NDict(item_data)

        # Run it through all the ops
        for op_kwarg_pair in self.all_ops_and_kwargs:
            input_ndict = op_kwarg_pair[0](input_ndict, **op_kwarg_pair[1])

        return input_ndict

    # FUNCTIONS TO BE IMPLEMENTED BY THE SUBCLASS

    # Since this is not a streaming dataset, the subclasses need to define how their data is read, and how a single raw sample is passed onto the superclass which will featurize it
    # Simplest case of this would be to do pd.read_csv() on the data file and then just return iloc[idx]
    def next_item(self, idx):
        raise NotImplementedError

    def get_ops_pipeline() -> PipelineDefault:
        raise NotImplementedError

    def collate_fn(self, batch):
        return batch
