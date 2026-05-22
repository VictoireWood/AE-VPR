import torch
import logging
import shutil
from collections import OrderedDict

import parser
args = parser.parse_arguments()

# def save_checkpoint(state, is_best, is_best_recall_25, is_best_recall_50, output_folder, ckpt_filename="last_checkpoint.pth"):
def save_checkpoint(state, is_best, output_folder, ckpt_filename="last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            'model_state_dict': state["model_state_dict"],
            'classifier_state_dict':state['classifier_state_dict'],
        }, f"{output_folder}/best_model.pth")

def save_checkpoint_with_groups(state, is_best, output_folder, ckpt_filename="last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            'model_state_dict': state["model_state_dict"],
            'classifiers_state_dict':state['classifiers_state_dict']
        }, f"{output_folder}/best_model.pth")

def save_checkpoint_with_groups_best_val(state, is_best, is_best_val, output_folder, ckpt_filename="last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            'model_state_dict': state["model_state_dict"],
            'classifiers_state_dict':state['classifiers_state_dict']
        }, f"{output_folder}/best_model.pth")

    if is_best_val:
        torch.save({
            'model_state_dict': state["model_state_dict"],
            'classifiers_state_dict':state['classifiers_state_dict']
        }, f"{output_folder}/best_val.pth")

# def resume_model(model: torch.nn.Module):
#     logging.info(f"Resuming model from {args.resume_model}")
#     checkpoint = torch.load(args.resume_model)
#     model_state_dict = checkpoint['model_state_dict']
#     if list(model_state_dict.keys())[0].startswith('module'):
#         model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
#     device = args.device
#     model.load_state_dict(model_state_dict)
#     model = model.to(device)
#     return model


def resume_model(model, classifier):
    logging.info(f"Resuming model from {args.resume_model}")
    checkpoint = torch.load(args.resume_model)

    model_state_dict = checkpoint["model_state_dict"]
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)

    classifier_state_dict = checkpoint["classifier_state_dict"]
    classifier.load_state_dict(classifier_state_dict)
    classifier = classifier.cpu()

    return model, classifier

def resume_model_with_classifiers(model, classifiers):
    logging.info(f"Resuming model from {args.resume_model}")
    checkpoint = torch.load(args.resume_model)

    model_state_dict = checkpoint["model_state_dict"]
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)

    assert len(classifiers) == len(checkpoint["classifiers_state_dict"]), \
        f"{len(classifiers)}, {len(checkpoint['classifiers_state_dict'])}"

    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
        c = c.cpu()

    return model, classifiers


def resume_train_with_params(output_folder, model, model_optimizer, classifier, classifier_optimizer):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]
    model_state_dict = checkpoint["model_state_dict"]
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)
    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    classifier.to(args.device)
    classifier_state_dict = checkpoint['classifier_state_dict']
    classifier.load_state_dict(classifier_state_dict)
    classifier_optimizer_state_dict = checkpoint['classifier_optimizer_state_dict']
    classifier_optimizer.load_state_dict(classifier_optimizer_state_dict)
    classifier = classifier.cpu()
    best_train_loss = checkpoint["best_train_loss"]
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)
    return model, model_optimizer, classifier, classifier_optimizer, best_train_loss, start_epoch_num

    
def resume_train_with_groups(output_folder, model, model_optimizer, classifiers, classifiers_optimizers):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]

    model_state_dict = checkpoint["model_state_dict"]
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)

    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert len(classifiers) == len(classifiers_optimizers) == len(
        checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        f"{len(classifiers)} , {len(classifiers_optimizers)} , {len(checkpoint['classifiers_state_dict'])} , {len(checkpoint['optimizers_state_dict'])}"

    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()
    best_train_loss = checkpoint["best_train_loss"]
    # Copy best model to current output_folder
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)

    return model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num


def resume_train_with_groups_all(output_folder, model, model_optimizer, classifiers, classifiers_optimizers, scheduler):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]

    model_state_dict = checkpoint["model_state_dict"]
    if list(model_state_dict.keys())[0].startswith('module'):
        model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
    model.load_state_dict(model_state_dict)

    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    assert len(classifiers) == len(classifiers_optimizers) == len(
        checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        f"{len(classifiers)} , {len(classifiers_optimizers)} , {len(checkpoint['classifiers_state_dict'])} , {len(checkpoint['optimizers_state_dict'])}"

    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()
    best_train_loss = checkpoint["best_train_loss"]
    best_val_lr = checkpoint['best_val_lr']
    # Copy best model to current output_folder
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)

    return model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num, scheduler, best_val_lr

# def resume_model(args, model, classifiers):
#     logging.info(f"Resuming model from {args.resume_model}")
#     checkpoint = torch.load(args.resume_model)

#     model_state_dict = checkpoint["model_state_dict"]
#     if list(model_state_dict.keys())[0].startswith('module'):
#         model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in model_state_dict.items()})
#     model.load_state_dict(model_state_dict)

#     assert len(classifiers) == len(checkpoint["classifiers_state_dict"]), \
#         f"{len(classifiers)}, {len(checkpoint['classifiers_state_dict'])}"

#     for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
#         # Move classifiers to GPU before loading their optimizers
#         c = c.to(args.device)
#         c.load_state_dict(sd)
#         c = c.cpu()

#     return model, classifiers