"""Serial clean/attack driver for frozen 30-frame official TraSw segments."""

import argparse
import json
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.evaluators import MOTEvaluator


def make_parser():
    parser = argparse.ArgumentParser("Frozen TraSw paired segment run")
    parser.add_argument("--mode", required=True, choices=["clean", "attack"])
    parser.add_argument("--rep", default="r1")
    parser.add_argument("--seg", type=int, default=None)
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--exp_file", default="exps/example/mot/yolox_x_mot17_half.py")
    parser.add_argument("-c", "--ckpt", default="pretrained/bytetrack_x_mot17.pth.tar")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("--conf", default=0.01, type=float)
    parser.add_argument("--nms", default=0.7, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100)
    parser.add_argument("--mot20", action="store_true", default=False)
    parser.add_argument("--attack", default="single")
    parser.add_argument("--method", default="ids")
    parser.add_argument("--img_dir", default="datasets/mot/train")
    parser.add_argument("--attack_id", type=int, default=-1)
    parser.add_argument("--output_dir", default="runs/_attack_stats")
    parser.add_argument("--rand", action="store_true", default=False)
    return parser


def main():
    args = make_parser().parse_args()
    if args.mode == "clean":
        args.attack = ""
        result_folder = os.path.join(args.outdir, "val_clean")
    else:
        args.attack = "single"
        args.method = "ids"
        args.attack_id = -1
        result_folder = os.path.join(args.outdir, "val_attack")
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.output_dir):
        if filename.endswith("_attack_result.txt"):
            os.remove(os.path.join(args.output_dir, filename))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("seeded testing")
    cudnn.benchmark = True
    exp = get_exp(args.exp_file, None)
    if args.seg is not None:
        ann_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __import__("yolox").__file__))), "datasets", "mot", "annotations")
        with open(os.path.join(ann_dir, exp.val_ann), encoding="utf-8") as handle:
            source = json.load(handle)
        images = [image for image in source["images"] if image["video_id"] == args.seg]
        assert images, "no images for seg %d" % args.seg
        image_ids = {image["id"] for image in images}
        source["images"] = images
        source["annotations"] = [item for item in source["annotations"]
                                 if item["image_id"] in image_ids]
        temporary = "_e5_2_seg.json"
        with open(os.path.join(ann_dir, temporary), "w", encoding="utf-8") as handle:
            json.dump(source, handle)
        exp.val_ann = temporary
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    model = exp.get_model()
    val_loader = exp.get_eval_loader(args.batch_size, False, False)
    evaluator = MOTEvaluator(args=args, dataloader=val_loader, img_size=exp.test_size,
                             confthre=exp.test_conf, nmsthre=exp.nmsthre,
                             num_classes=exp.num_classes)
    torch.cuda.set_device(0)
    model.cuda(0)
    model.eval()
    checkpoint = torch.load(args.ckpt, map_location="cuda:0")
    model.load_state_dict(checkpoint["model"])
    model = fuse_model(model)
    try:
        evaluator.evaluate(model, False, True, None, None, exp.test_size, result_folder)
    except RuntimeError as error:
        if str(error) != "Finish":
            raise
    print("TRASW_DONE mode=%s rep=%s result_folder=%s" %
          (args.mode, args.rep, result_folder))


if __name__ == "__main__":
    main()
