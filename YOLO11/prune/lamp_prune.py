import torch
import argparse
from ultralytics import YOLO
import torch.nn.utils.prune as prune
from tqdm import tqdm

def compute_lamp_scores(weight: torch.Tensor):
    """
    Compute LAMP scores for a weight tensor (same math as the paper).
    Returns a tensor of same shape with scores (on the same device as weight).
    """
    # work on CPU to save GPU memory
    w = weight.detach().cpu().flatten()
    w_abs2 = (w.abs() ** 2)
    vals, indices = torch.sort(w_abs2)  # ascending
    # suffix sum: denom for each sorted position
    suffix = torch.flip(torch.cumsum(torch.flip(vals, dims=[0]), dim=0), dims=[0])
    lamp_sorted = vals / (suffix + 1e-12)
    lamp = torch.empty_like(lamp_sorted)
    lamp[indices] = lamp_sorted
    return lamp.reshape(weight.shape).to(weight.device)

def global_lamp_prune(model_wrapper, global_sparsity=0.8, prunable_types=(torch.nn.Conv2d, torch.nn.Linear),
                      exclude_first_conv=True, dry_run=False):
    """
    model_wrapper: Ultralytics YOLO wrapper (YOLO(...))
    global_sparsity: fraction of weights to remove (e.g. 0.8 means keep 20%)
    exclude_first_conv: skip pruning the very first conv encountered (safer)
    If dry_run=True, only print statistics and do not modify model.
    Returns dict summary.
    """
    net = model_wrapper.model  # underlying nn.Module
    device = next(net.parameters()).device if any(p.requires_grad for p in net.parameters()) else torch.device("cpu")

    # collect scores
    all_scores = []
    modules = []
    first_conv_skipped = False
    print("Collecting LAMP scores (CPU)...")
    for name, module in net.named_modules():
        if isinstance(module, prunable_types) and hasattr(module, "weight"):
            if exclude_first_conv and (not first_conv_skipped):
                # skip the first conv
                first_conv_skipped = True
                print(f"Skipping first prunable layer: {name}")
                continue
            w = module.weight
            lamp = compute_lamp_scores(w)
            all_scores.append(lamp.flatten().cpu())
            modules.append((module, name, lamp.shape))
    all_scores_flat = torch.cat(all_scores)
    total_params = all_scores_flat.numel()
    k_keep = int(total_params * (1.0 - global_sparsity))
    if k_keep < 1:
        raise ValueError("global_sparsity too large -> no params left to keep.")
    topk_vals, _ = torch.topk(all_scores_flat, k_keep)
    threshold = topk_vals.min().item()
    print(f"Total params considered: {total_params}, keep top {k_keep} -> threshold {threshold:.6e}")

    # build per-layer masks
    masks = {}
    keep_counts = {}
    for module, name, shape in modules:
        lamp = compute_lamp_scores(module.weight)
        mask = (lamp >= threshold).to(torch.uint8)  # binary mask
        # ensure at least one weight kept per layer
        if mask.sum().item() == 0:
            # keep the max-scoring weight
            flat = lamp.flatten()
            _, idx = torch.max(flat, dim=0)
            mask_flat = torch.zeros_like(flat, dtype=torch.uint8)
            mask_flat[idx] = 1
            mask = mask_flat.reshape(shape).to(torch.uint8)
        masks[id(module)] = (module, mask)
        keep_counts[name] = int(mask.sum().item())

    # summary
    total_kept = sum(keep_counts.values())
    print("Pruning summary (per layer keep counts):")
    for n, k in keep_counts.items():
        print(f"  {n:50s}  keep {k}")

    print(f"Global kept weights: {total_kept} / {total_params}  -> actual sparsity = {1 - total_kept/total_params:.4f}")

    if dry_run:
        return {"total_params": total_params, "kept": total_kept, "keep_counts": keep_counts}

    # apply masks (use torch.nn.utils.prune custom_from_mask then remove the parametrization)
    print("Applying masks to model (this will modify model weights)...")
    for (module, mask) in masks.values():
        # mask should be same dtype as weight for custom_from_mask
        mask_t = mask.to(dtype=module.weight.dtype, device=module.weight.device)
        prune.custom_from_mask(module, "weight", mask_t)
        # remove param reparam so model.state_dict() contains pruned weights (not masked param)
        prune.remove(module, "weight")

    return {"total_params": total_params, "kept": total_kept, "keep_counts": keep_counts, "threshold": threshold}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to original .pt checkpoint")
    parser.add_argument("--out", type=str, required=True, help="path to save pruned .pt")
    parser.add_argument("--sparsity", type=float, default=0.8, help="fraction of weights to remove (0..1)")
    parser.add_argument("--dry", action="store_true", help="only compute stats, don't modify model")
    args = parser.parse_args()

    print("Loading model:", args.weights)
    y = YOLO(args.weights)
    # move model to CPU for pruning ops
    try:
        y.model.cpu()
    except Exception:
        pass

    summary = global_lamp_prune(y, global_sparsity=args.sparsity, dry_run=args.dry)
    if args.dry:
        print("Dry run complete. Summary:", summary)
        exit(0)

    # save pruned checkpoint (Ultralytics expects a dict with 'model' key)
    ckpt = {"model": y.model.state_dict(), "pruned": True, "prune_meta": {"method": "LAMP", "sparsity": args.sparsity}}
    print("Saving pruned checkpoint to:", args.out)
    torch.save(ckpt, args.out)
    print("Saved. You can now fine-tune this checkpoint with Ultralytics training (see README instructions).")
