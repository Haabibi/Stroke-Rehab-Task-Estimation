#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from model_loader import load_checkpoint, make_model

from PIL import Image 
import numpy as np
from torchvision import transforms 

from class_parse import parse_verb, parse_noun 

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model_type",
    nargs="?",
    choices=["tsn", "tsm", "tsm-nl", "trn", "mtrn"],
    default=None,
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Path to checkpointed model. Should be a dictionary containing the keys:"
    " 'model_type', 'segment_count', 'modality', 'state_dict', and 'arch'.",
)
parser.add_argument(
    "--arch",
    default="resnet50",
    choices=["BNInception", "resnet50"],
    help="Backbone architecture",
)
parser.add_argument(
    "--modality", default="RGB", choices=["RGB", "Flow"], help="Input modality"
)
parser.add_argument(
    "--flow-length", default=5, type=int, help="Number of (u, v) pairs in flow stack"
)
parser.add_argument(
    "--dropout",
    default=0.7,
    type=float,
    help="Dropout probability. The dropout layer replaces the "
    "backbone's classification layer.",
)
parser.add_argument(
    "--trn-img-feature-dim",
    default=256,
    type=int,
    help="Number of dimensions for the output of backbone network. "
    "This is effectively the image feature dimensionality.",
)
parser.add_argument(
    "--segment-count",
    default=8,
    type=int,
    help="Number of segments. For RGB this corresponds to number of "
    "frames, whereas for Flow, it is the number of points from "
    "which a stack of (u, v) frames are sampled.",
)
parser.add_argument(
    "--tsn-consensus-type",
    choices=["avg", "max"],
    default="avg",
    help="Consensus function for TSN used to fuse class scores from "
    "each segment's predictoin.",
)
parser.add_argument(
    "--tsm-shift-div",
    default=8,
    type=int,
    help="Reciprocal proportion of features temporally-shifted.",
)
parser.add_argument(
    "--tsm-shift-place",
    default="blockres",
    choices=["block", "blockres"],
    help="Location for the temporal shift to take place. Either 'block' for the shift "
    "to happen in the non-residual part of a block, or 'blockres' if the shift happens "
    "in the residual path.",
)
parser.add_argument(
    "--tsm-temporal-pool",
    action="store_true",
    help="Gradually temporally pool throughout the network",
)
parser.add_argument("--batch-size", default=1, type=int, help="Batch size for demo")
parser.add_argument("--print-model", action="store_true", help="Print model definition")


def extract_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    settings = vars(args)
    for variant in ["trn", "tsm", "tsn"]:
        variant_key_prefix = f"{variant}_"
        variant_keys = {
            key for key in settings.keys() if key.startswith(variant_key_prefix)
        }
        for key in variant_keys:
            stripped_key = key[len(variant_key_prefix) :]
            settings[stripped_key] = settings[key]
            del settings[key]
    return settings


def main(args):
    logging.basicConfig(level=logging.INFO)
    if args.checkpoint is None:
        if args.model_type is None:
            print("If not providing a checkpoint, you must specify model_type")
            sys.exit(1)
        settings = extract_settings_from_args(args)
        model = make_model(settings)
    elif args.checkpoint is not None and args.checkpoint.exists():
        model = load_checkpoint(args.checkpoint)
    else:
        print(f"{args.checkpoint} doesn't exist")
        sys.exit(1)

    if args.print_model:
        print(model)
    height, width = model.input_size, model.input_size
    if model.modality == "RGB":
        channel_dim = 3
    elif model.modality == "Flow":
        channel_dim = args.flow_length * 2
    else:
        raise ValueError(f"Unknown modality {args.modality}")

    model.eval()

    #print("IS MODEL TRAINING?", model.training)
    """
    input = torch.randn(
        [args.batch_size, args.segment_count, channel_dim, height, width]
    )
    """
    
    #print(f"DEBUG: height: - {height}, width - {width}") 
    path = "survey_imgs/"
    dirs = sorted(os.listdir(path))
    for file in dirs:
        print("THIS IS FILE: ", file)
        image = Image.open(os.path.join(path, file))
        imsize = (height, width)
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = loader(image).float()
        image = image.unsqueeze(0)
        images = image.repeat(8, 1, 1, 1)
        
        print(f"DEBUG2: IMAGE SHAPE {image.shape} IMAGES SHAPE {images.shape}")
        
        #print(f"Input shape: {images.shape}")
        # Models take input in the format
        # [n_batch, n_segments, n_channels, height, width]
        # output = model(input)

        output = model(images)
        if isinstance(output, tuple):
            #print(f"Output shape: {[o.shape for o in output]}")
            act_rst = max(output[0].squeeze(0))
            noun_rst = max(output[1].squeeze(0))
            #print(act_rst, noun_rst)
            #print((output[0].squeeze(0) == act_rst).nonzero())
            #print((output[1].squeeze(0) == noun_rst).nonzero())
            
            action = parse_verb()[int((output[0].squeeze(0) == act_rst).nonzero()[0][0])]
            noun = parse_noun()[int((output[1].squeeze(0) == noun_rst).nonzero()[0][0])] 

            print(action, noun)
        else:
            print(f"Output shape: {output.shape}")
        


if __name__ == "__main__":
    main(parser.parse_args())
