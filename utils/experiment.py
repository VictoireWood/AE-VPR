from __future__ import annotations

from torch.utils.data import ConcatDataset

from models.classifiers import AAMC, LMCC, QAMC, LinearLayer


def get_dataset_split_names(args):
    dataset_name = args.dataset_name
    if dataset_name in {"ct01", "ct02"}:
        train_dataset_folders = [dataset_name]
        default_test_datasets = [dataset_name]
    elif dataset_name == "2022" or "2022" in dataset_name:
        train_dataset_folders = [dataset_name]
        default_test_datasets = ["new_photo"]
    else:
        train_dataset_folders = [dataset_name]
        default_test_datasets = ["new_photo"]

    test_datasets = args.test_set_list if args.test_set_list is not None else default_test_datasets
    if isinstance(test_datasets, str):
        test_datasets = [test_datasets]
    return train_dataset_folders, list(test_datasets)


def make_backbone_info(args):
    backbone = args.backbone.lower()
    if "dinov2" in backbone:
        return {
            "input_size": args.train_resize,
            "num_trainable_blocks": args.num_trainable_blocks,
        }
    if "efficientnet" in backbone:
        return {
            "input_size": args.train_resize,
            "layers_to_freeze": args.layers_to_freeze,
        }
    if "resnet" in backbone:
        return {
            "input_size": args.train_resize,
            "layers_to_freeze": args.layers_to_freeze,
            "layers_to_crop": list(args.layers_to_crop),
        }
    if "swin" in backbone:
        return {
            "input_size": args.train_resize,
            "layers_to_freeze": args.layers_to_freeze,
        }
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def build_test_dataset(args, test_datasets, test_transform):
    from dataloaders.HCDataset import TestDataset, TestDatasetNew, realHCDataset_N

    datasets = []
    dataframe_backed_sets = []

    for dataset_name in test_datasets:
        if dataset_name == "real_photo":
            if args.val_set_path is None:
                raise ValueError("--val_set_path is required for real_photo evaluation")
            datasets.append(realHCDataset_N(base_path=args.val_set_path, M=args.M, N=args.N, transform=test_transform))
        elif dataset_name == "new_photo" or "qd_test" in dataset_name:
            if args.test_set_path is None:
                raise ValueError("--test_set_path is required for new_photo/qd_test evaluation")
            datasets.append(TestDatasetNew(test_folder=args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize))
        else:
            dataframe_backed_sets.append(dataset_name)

    if dataframe_backed_sets:
        if args.test_set_path is None:
            raise ValueError("--test_set_path is required for dataframe-backed test datasets")
        datasets.append(
            TestDataset(
                test_folder=args.test_set_path,
                test_datasets=dataframe_backed_sets,
                M=args.M,
                N=args.N,
                image_size=args.test_resize,
            )
        )

    if not datasets:
        raise ValueError("No test dataset was built; check --test_set_list")
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def build_classifiers(args, feature_dim, groups):
    if args.classifier == "AAMC":
        return [AAMC(in_features=feature_dim, out_features=group.get_classes_num(), s=args.aamc_s, m=args.aamc_m) for group in groups]
    if args.classifier == "QAMC":
        return [QAMC(embedding_size=feature_dim, classnum=group.get_classes_num(), m=args.aamc_m, s=args.aamc_s) for group in groups]
    if args.classifier == "LMCC":
        return [LMCC(embedding_size=feature_dim, classnum=group.get_classes_num(), m=args.aamc_m, s=args.aamc_s) for group in groups]
    if args.classifier == "LinearLayer":
        return [LinearLayer(embedding_size=feature_dim, classnum=group.get_classes_num()) for group in groups]
    raise ValueError(f"Unsupported classifier: {args.classifier}")


def classifier_forward(classifier, descriptors, labels, images=None):
    if isinstance(classifier, QAMC):
        return classifier(descriptors, labels, images=images)
    return classifier(descriptors, labels)
