from enum import Enum

FIELD_INDEX = "index"

FIELD_DATA = "data"
FIELD_INPUT = "input"
FIELD_DATA_INPUT = ".".join([FIELD_DATA, FIELD_INPUT])
FIELD_DATA_LIGAND = ".".join([FIELD_DATA_INPUT, "ligand"])
FIELD_DATA_PROTEIN = ".".join([FIELD_DATA_INPUT, "protein"])
FIELD_DATA_PROTEIN_EMB = ".".join([FIELD_DATA_PROTEIN, "protein_embedding"])

FIELD_SMILES = "smiles"
FIELD_PROTEIN = "protein"
FIELD_IMAGE = "img"
FIELD_TOKENIZED_SMILES = ".".join([FIELD_SMILES, "tokenized"])

FIELD_PATHS_IMAGE = ".".join(["paths", "img"])
FIELD_DATA_LIGAND_SMILES = ".".join([FIELD_DATA_LIGAND, FIELD_SMILES])
FIELD_DATA_SAMPLE_ID = ".".join([FIELD_DATA, "sample_id"])
FIELD_DATA_LIGAND_IMAGE = ".".join([FIELD_DATA_LIGAND, FIELD_IMAGE])
FIELD_DATA_LIGAND_FINGERPRINT = ".".join([FIELD_DATA_LIGAND, "rdkit_fingerprint"])
FIELD_DATA_LIGAND_JIGSAW_IMG = ".".join([FIELD_DATA_LIGAND, "jigsawed_img"])
FIELD_DATA_LIGAND_AUG_MASK_IMG = ".".join([FIELD_DATA_LIGAND, "augmented_masked_img"])
FIELD_DATA_LIGAND_AUG_NONMASK_IMG = ".".join(
    [FIELD_DATA_LIGAND, "augmented_nonmasked_img"]
)
FIELD_DATA_LIGAND_AUG_NONMASK_IMG_SMALL = ".".join(
    [FIELD_DATA_LIGAND, "augmented_nonmasked_img_small"]
)
FIELD_DATA_LIGAND_AUG_IMG_TEMP = ".".join([FIELD_DATA_LIGAND, "augmented_img_temp"])
LABEL_JIGSAW = "jigsaw_order"

FIELD_LABELS = "labels"
FIELD_LABEL = "label"
FIELD_DATA_LABELS = ".".join([FIELD_DATA_INPUT, FIELD_LABELS])
FIELD_DATA_CLUSTER = ".".join([FIELD_DATA_LABELS, "cluster"])
FIELD_DATA_CLUSTER_ALL = ".".join([FIELD_DATA_CLUSTER, "all_clusters"])
FIELD_DATA_JIGSAW_ORDER = ".".join([FIELD_DATA_LABELS, LABEL_JIGSAW])

FIELD_MLM = "mlm"
FIELD_DATA_LIGAND_MLM = ".".join([FIELD_DATA_LIGAND, FIELD_MLM])
FIELD_DATA_LIGAND_MLM_INPUT = ".".join([FIELD_DATA_LIGAND_MLM, "input"])
FIELD_DATA_LIGAND_MLM_LABEL = ".".join([FIELD_DATA_LIGAND_MLM, "label"])

FIELD_GRAPH = "graph"
FIELD_GRAPH2D = "graph2d"
FIELD_GRAPH2D_GNN = "graph2d_gnn"
FIELD_LABELS_GRAPH = ".".join([FIELD_LABELS, FIELD_GRAPH])
FIELD_LABELS_GRAPH_DATA = ".".join([FIELD_LABELS_GRAPH, FIELD_DATA])


class AutoConvertStrEnum(Enum):
    def __call__(cls, value):
        if isinstance(value, str) and value.isdigit():
            return cls(int(value))
        elif isinstance(value, int) and value.isdigit():
            return cls(value)
        elif isinstance(value, cls):
            return value
        raise ValueError(f"Invalid input: {value}")


class Modality(AutoConvertStrEnum):
    IMAGE = 1
    TEXT = 2
    GRAPH = 3
    MULTIVIEW = 4
    GRAPH_3D = 5

    @staticmethod
    def from_str(label):
        if label in ("IMAGE", "image"):
            return Modality.IMAGE
        elif label in ("TEXT", "text"):
            return Modality.TEXT
        elif label in ("GRAPH", "graph"):
            return Modality.GRAPH
        elif label in ("MULTIVIEW", "multiview"):
            return Modality.MULTIVIEW
        elif label in ("GRAPH3D", "graph3d"):
            return Modality.GRAPH_3D


class Models(Enum):
    IMAGE_MODEL = (
        "bmfm_sm.predictive.modules.image_models.ImageModel",
        "image",
        Modality.IMAGE,
        True,
    )
    TEXT_MODEL = (
        "bmfm_sm.predictive.modules.text_models.TextModel",
        "text",
        Modality.TEXT,
        True,
    )
    GRAPH_GCN_MODEL = (
        "bmfm_sm.predictive.modules.graph_2d_models.GCNModel",
        "graph_gcn",
        Modality.GRAPH,
        False,
    )
    GRAPH_GIN_MODEL = (
        "bmfm_sm.predictive.modules.graph_2d_models.GINModel",
        "graph_gin",
        Modality.GRAPH,
        False,
    )
    GRAPH_ATTENTIVEFP_MODEL = (
        "bmfm_sm.predictive.modules.graph_2d_models.AttentiveFPModel",
        "graph_attentivefp",
        Modality.GRAPH,
        False,
    )
    GRAPH_TRIMNET_MODEL = (
        "bmfm_sm.predictive.modules.graph_2d_models.TrimNetModel",
        "graph_trimnet",
        Modality.GRAPH,
        False,
    )
    GRAPH_2D_MODEL = (
        "bmfm_sm.predictive.modules.graph_2d_models.Graph2dModel",
        "graph_2d",
        Modality.GRAPH,
        True,
    )
    GRAPH_3D_MODEL = (
        "bmfm_sm.predictive.modules.graph_3d_models.Graph3dMPPModel",
        "graph_3d",
        Modality.GRAPH_3D,
        True,
    )
    SMMV_MODEL = (
        "bmfm_sm.predictive.modules.smmv_model.SmallMoleculeMultiView",
        "smmv",
        Modality.MULTIVIEW,
        True,
    )

    def __init__(self, cls, short_name, modality: Modality, pretrained=False):
        self.cls = cls
        self.short_name = short_name
        self.modality = modality
        self.pretrained = pretrained


class LateFusionStrategy(Enum):
    ATTENTIONAL = ("coeff_mlp", "coeff_mlp", None)
    ATTENTIONAL_SIMPLE = ("coeff", "coeff", None)
    CONCAT = ("concat", "concat", None)

    MOE_WEIGHTED_CONCAT_PROJECTED = ("moe_proj", "moe_weighted_concat", "projected")
    MOE_WEIGHTED_CONCAT_UNPROJECTED = (
        "moe_unproj",
        "moe_weighted_concat",
        "unprojected",
    )
    MOE_WEIGHTED_CONCAT_BOTH_PROJECTED = (
        "moe_both_proj",
        "moe_weighted_concat",
        "both_projected",
    )

    MOE_NOISED_WEIGHTED_CONCAT_PROJECTED = (
        "moe_noised_proj",
        "moe_noised_weighted_concat",
        "projected",
    )
    MOE_NOISED_WEIGHTED_CONCAT_UNPROJECTED = (
        "moe_noised_unproj",
        "moe_noised_weighted_concat",
        "unprojected",
    )
    MOE_NOISED_WEIGHTED_CONCAT_BOTH_PROJECTED = (
        "moe_noised_both_proj",
        "moe_noised_weighted_concat",
        "both_projected",
    )

    def __init__(self, short_name, agg_arch, projection_type):
        self.short_name = short_name
        self.agg_arch = agg_arch
        self.projection_type = projection_type

    @classmethod
    def from_string(cls, value):
        # Search for the enum by matching the name, case-insensitive
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Metrics(AutoConvertStrEnum):
    ROCAUC = "rocauc"
    RMSE = "rmse"
    MAE = "mae"


class SplitStrategy(Enum):
    NONE = "none"
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRATIFIED = "stratified"
    LIGAND_SCAFFOLD = "ligand_scaffold"
    LIGAND_RANDOM_SCAFFOLD = "ligand_random_scaffold"
    LIGAND_SCAFFOLD_BALANCED = "ligand_scaffold_balanced"
    CUSTOM = "custom"


ENV_RANDOM_SEED = "PL_GLOBAL_SEED"
