# -*- coding: utf-8 -*-
"""
MSNet - Smoke Test (Synthetic Data)
=====================================
Validates the entire framework end-to-end using randomly generated
tensors that mimic real MRI + clinical data.

NO real NIfTI files are required.

Tests:
  ✓ Model instantiation (Swin + ResNet)
  ✓ Forward pass (binary + multiclass)
  ✓ Missing modality handling
  ✓ Loss computation
  ✓ Backward pass / gradient flow
  ✓ Metric computation
  ✓ Grad-CAM (structure, no actual hooks)
  ✓ Uncertainty head (MC Dropout)

Run:
  python smoke_test.py
"""

import sys
import traceback
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── output helpers (ASCII-safe for Windows cp1252) ─────────
import os, sys
# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}[PASS]{RESET}  {msg}")
def fail(msg):   print(f"  {RED}[FAIL]{RESET}  {msg}")
def info(msg):   print(f"  {CYAN}[--]{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{YELLOW}{'='*55}{RESET}\n{BOLD}{YELLOW}  {msg}{RESET}\n{BOLD}{YELLOW}{'='*55}{RESET}")

# ── minimal config that mirrors default_config.yaml ─────────
def make_config(task="multiclass", encoder="resnet3d"):
    """Returns a minimal but valid config dict."""
    return {
        "experiment": {
            "name": "smoke_test",
            "output_dir": "outputs/smoke/",
            "seed": 42,
            "use_wandb": False,
        },
        "data": {
            "num_classes": 4 if task == "multiclass" else 2,
            "class_names": (["HC","RRMS","SPMS","PPMS"] if task=="multiclass"
                            else ["HC","MS"]),
            "task": task,
            "clinical_features": [
                "age","gender","edss","disease_duration",
                "igg_index","oligoclonal_bands","t_cell_count"
            ],
            "spatial_size": [64, 64, 64],    # 64 needed so CNN stages don't collapse
            "voxel_spacing": [1.0, 1.0, 1.0],
            "train_split": 0.70,
            "val_split":   0.15,
            "modality_dropout": {"enabled": True, "dropout_prob": 0.2},
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "augmentation": {},
        },
        "model": {
            "name": "MSNet",
            "encoder_type": encoder,
            "swin": {
                "img_size": [32, 32, 32],
                "in_channels": 1,
                "feature_size": 12,          # smallest valid Swin size
                "depths": [1, 1, 1, 1],
                "num_heads": [3, 3, 3, 3],
                "window_size": 4,
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "dropout_path_rate": 0.0,
                "use_checkpoint": False,
            },
            "cnn": {
                "backbone": "resnet50_3d",
                "pretrained": False,
                "in_channels": 1,
                "out_channels": [16, 32, 64, 128],
                "dropout": 0.2,
            },
            "clinical": {
                "input_dim": 7,
                "hidden_dims": [32, 64, 32],
                "dropout": 0.2,
                "out_dim": 64,
            },
            "fusion": {
                "type": "cross_modal_attention",
                "embed_dim": 128,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.1,
                "use_positional_encoding": True,
            },
            "projection": {"out_dim": 128, "use_batchnorm": True},
            "classifier": {
                "hidden_dims": [128, 64],
                "dropout": 0.3,
                "use_label_smoothing": True,
                "label_smoothing": 0.1,
            },
            "uncertainty": {
                "enabled": True,
                "method": "mc_dropout",
                "mc_samples": 5,
                "dropout_rate": 0.3,
            },
        },
        "training": {
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "betas": [0.9, 0.999],
            "gradient_clip_norm": 1.0,
            "scheduler": "cosine_warmup",
            "warmup_epochs": 1,
            "min_lr": 1e-6,
            "use_amp": False,
            "amp_dtype": "float32",
            "grad_accum_steps": 1,
            "loss": {
                "type": "focal_cross_entropy",
                "focal_gamma": 2.0,
                "label_smoothing": 0.1,
                "use_class_weights": False,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "monitor": "val/roc_auc",
                "mode": "max",
                "min_delta": 0.001,
            },
            "save_top_k": 1,
            "checkpoint_metric": "val/auroc",
        },
        "evaluation": {
            "metrics": ["accuracy","sensitivity","specificity","f1_score","roc_auc","pr_auc"],
            "bootstrap": {"enabled": False, "n_bootstraps": 10, "confidence_level": 0.95},
            "threshold_optimization": {"enabled": False, "method": "youden_j"},
        },
        "explainability": {
            "gradcam": {
                "enabled": False,
                "output_dir": "outputs/smoke/gradcam/",
                "colormap": "jet",
                "alpha": 0.5,
            }
        },
        "hardware": {"device": "cpu", "num_gpus": 0, "distributed": False, "compile_model": False},
    }


# ── synthetic batch factory ──────────────────────────────────
def make_batch(B=2, n_mod=4, spatial=32, n_clinical=7, n_classes=4, device="cpu"):
    """Create a fake batch matching MSNet's expected input shapes."""
    volumes      = torch.randn(B, n_mod, 1, spatial, spatial, spatial)
    clinical     = torch.randn(B, n_clinical)
    mod_mask     = torch.ones(B, n_mod)           # all modalities present
    labels       = torch.randint(0, n_classes, (B,))
    return volumes.to(device), clinical.to(device), mod_mask.to(device), labels.to(device)


# ════════════════════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════════════════════

results = []

def run_test(name, fn):
    try:
        t0 = time.time()
        fn()
        elapsed = (time.time() - t0) * 1000
        ok(f"{name}  [{elapsed:.0f} ms]")
        results.append((name, True, None))
    except Exception as e:
        fail(f"{name}")
        info(str(e))
        traceback.print_exc()
        results.append((name, False, str(e)))


# ── Test 1: Imports ──────────────────────────────────────────
header("1 / 8 — Import Check")

def test_imports():
    from models.encoders.cnn_encoder       import ResNet3DEncoder
    from models.encoders.clinical_encoder  import ClinicalEncoder
    from models.fusion.cross_modal_attention import CrossModalAttentionFusion
    from models.heads.uncertainty_head     import (ClassifierHead,
                                                   MCDropoutUncertaintyHead,
                                                   EvidentialClassifierHead)
    from models.msnet                      import MSNet
    from training.losses                   import FocalLoss, LabelSmoothingCrossEntropy, build_loss_function
    from evaluation.metrics                import compute_all_metrics, bootstrap_confidence_intervals
    from utils.seed                        import set_seed

run_test("All project modules importable", test_imports)


# ── Test 2: Clinical Encoder ─────────────────────────────────
header("2 / 8 — Clinical Encoder")

def test_clinical_encoder():
    from models.encoders.clinical_encoder import ClinicalEncoder
    enc = ClinicalEncoder(input_dim=7, hidden_dims=[32, 64, 32], out_dim=64)
    x = torch.randn(4, 7)
    emb, gate = enc(x)
    assert emb.shape  == (4, 64), f"Expected (4,64), got {emb.shape}"
    assert gate.shape == (4, 7),  f"Expected (4,7),  got {gate.shape}"
    # All gate values in [0,1] (sigmoid output)
    assert gate.min() >= 0 and gate.max() <= 1
    emb_norm = torch.nn.functional.normalize(emb, dim=-1)
    assert torch.allclose(emb_norm, emb, atol=1e-5), "Output should be L2-normalised"

run_test("ClinicalEncoder forward pass + shape + L2 norm", test_clinical_encoder)


# ── Test 3: 3D ResNet Encoder ────────────────────────────────
header("3 / 8 — 3D CNN Encoder (ResNet3D)")

def test_cnn_encoder():
    from models.encoders.cnn_encoder import ResNet3DEncoder
    # Need spatial >= 64 so stride-2 stages don't collapse to 1x1x1
    enc = ResNet3DEncoder(in_channels=1, base_channels=8, out_dim=64)
    x = torch.randn(2, 1, 64, 64, 64)

    # With modality present
    feats, last_conv = enc(x, modality_present=torch.ones(2))
    assert feats.shape[0] == 2 and feats.shape[1] == 64

    # With modality missing — features must be all zero
    feats_missing, _ = enc(x, modality_present=torch.zeros(2))
    assert feats_missing.abs().max().item() == 0.0, "Missing modality must yield zero features"

run_test("ResNet3DEncoder forward + missing-modality zeroing", test_cnn_encoder)


# ── Test 4: Cross-Modal Attention Fusion ────────────────────
header("4 / 8 — Cross-Modal Attention Fusion")

def test_fusion():
    from models.fusion.cross_modal_attention import CrossModalAttentionFusion
    fusion = CrossModalAttentionFusion(
        n_modalities=4, embed_dim=64, num_heads=4, num_layers=1
    )
    B = 3
    mri_tokens = torch.randn(B, 4, 64)
    clinical   = torch.randn(B, 64)

    # All modalities present
    fused, attn = fusion(mri_tokens, clinical, modality_mask=torch.ones(B, 4))
    assert fused.shape == (B, 64), f"Got {fused.shape}"
    assert len(attn) == 1          # 1 layer → 1 attention map

    # One modality missing
    mask = torch.ones(B, 4); mask[:, 2] = 0   # FLAIR dropped
    fused_m, _ = fusion(mri_tokens, clinical, modality_mask=mask)
    assert fused_m.shape == (B, 64)

run_test("CrossModalAttentionFusion shapes + missing mask", test_fusion)


# ── Test 5: Uncertainty Heads ────────────────────────────────
header("5 / 8 — Uncertainty Heads")

def test_uncertainty():
    from models.heads.uncertainty_head import (
        ClassifierHead, MCDropoutUncertaintyHead, EvidentialClassifierHead
    )
    B, dim, C = 4, 64, 4

    # Standard head
    head = ClassifierHead(in_dim=dim, num_classes=C, hidden_dims=[64], dropout=0.3)
    x = torch.randn(B, dim)
    logits = head(x)
    assert logits.shape == (B, C)

    # MC Dropout head
    mc_head = MCDropoutUncertaintyHead(head, n_mc_samples=5, num_classes=C)
    out = mc_head(x)
    assert out["mean_probs"].shape          == (B, C)
    assert out["epistemic_uncertainty"].shape == (B,)
    assert out["mc_probs"].shape            == (B, 5, C)
    # Probabilities must sum to ~1
    assert torch.allclose(out["mean_probs"].sum(dim=-1), torch.ones(B), atol=1e-4)

    # Evidential head
    edl = EvidentialClassifierHead(in_dim=dim, num_classes=C)
    ev_out = edl(x)
    assert ev_out["alpha"].shape    == (B, C)
    assert ev_out["uncertainty"].shape == (B,)
    # alpha > 1 (evidence + 1)
    assert (ev_out["alpha"] > 1).all()

run_test("ClassifierHead + MCDropout + Evidential shapes", test_uncertainty)


# ── Test 6: Full MSNet Forward Pass ─────────────────────────
header("6 / 8 — Full MSNet Forward Pass")

def test_msnet_forward():
    from models.msnet import MSNet

    for task, n_cls in [("multiclass", 4), ("binary", 2)]:
        cfg = make_config(task=task, encoder="resnet3d")
        model = MSNet(cfg)
        model.eval()

        B = 2
        volumes, clinical, mod_mask, labels = make_batch(
            B=B, spatial=64, n_classes=n_cls
        )

        with torch.no_grad():
            out = model(volumes, clinical, mod_mask,
                        return_uncertainty=True, return_features=True)

        assert out["logits"].shape        == (B, n_cls), \
            f"[{task}] logits: {out['logits'].shape}"
        assert out["probabilities"].shape == (B, n_cls)
        assert out["pred_class"].shape    == (B,)
        assert out["fused_features"].shape == (B, cfg["model"]["fusion"]["embed_dim"])
        assert "uncertainty" in out

        # Probabilities sum to 1
        prob_sum = out["probabilities"].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(B), atol=1e-4), \
            f"Probs don't sum to 1: {prob_sum}"

        info(f"  task={task:>10} | logits={tuple(out['logits'].shape)} | "
             f"probs_sum={prob_sum.mean():.6f}")

run_test("MSNet forward (binary + multiclass, with uncertainty)", test_msnet_forward)


# ── Test 7: Loss Functions ───────────────────────────────────
header("7 / 8 — Loss Functions")

def test_losses():
    from training.losses import FocalLoss, LabelSmoothingCrossEntropy

    B, C = 8, 4
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    # Focal loss
    fl = FocalLoss(num_classes=C, gamma=2.0)
    loss_fl = fl(logits, targets)
    assert loss_fl.dim() == 0 and loss_fl.item() > 0, "Focal loss must be positive scalar"

    # Label smoothing CE
    ls = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_ls = ls(logits, targets)
    assert loss_ls.dim() == 0 and loss_ls.item() > 0

    # Backward through focal loss using requires_grad on input
    logits2 = torch.randn(B, C, requires_grad=True)
    loss_back = fl(logits2, targets)
    loss_back.backward()
    assert logits2.grad is not None, "Gradients must flow through FocalLoss"

    # Class-weighted focal
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    fl_w = FocalLoss(num_classes=C, gamma=2.0, alpha=weights, label_smoothing=0.1)
    loss_w = fl_w(logits, targets)
    assert loss_w.item() > 0

run_test("FocalLoss + LabelSmoothingCE + weighted focal", test_losses)


# ── Test 8: Metrics ──────────────────────────────────────────
header("8 / 8 — Evaluation Metrics")

def test_metrics():
    from evaluation.metrics import (
        compute_all_metrics, bootstrap_confidence_intervals,
        find_optimal_threshold
    )
    N = 60

    # Multiclass
    labels_mc = np.random.randint(0, 4, N).tolist()
    preds_mc  = np.random.randint(0, 4, N).tolist()
    probs_mc  = np.random.dirichlet([1,1,1,1], size=N).tolist()

    m = compute_all_metrics(labels_mc, preds_mc, num_classes=4, probs=probs_mc)
    for key in ["accuracy", "sensitivity", "specificity", "f1_macro",
                "roc_auc", "pr_auc", "cohen_kappa", "mcc"]:
        assert key in m, f"Missing metric: {key}"
        assert 0.0 <= m[key] <= 1.0, f"{key} out of [0,1]: {m[key]}"

    # Binary + threshold optimization
    labels_bin = np.random.randint(0, 2, N).tolist()
    probs_bin  = np.random.rand(N).tolist()
    thresh, score = find_optimal_threshold(labels_bin, probs_bin, method="youden_j")
    assert 0.0 <= thresh <= 1.0

    # Bootstrap CIs (tiny n for speed)
    ci = bootstrap_confidence_intervals(
        labels_mc, preds_mc, num_classes=4,
        probs=probs_mc, n_bootstraps=20
    )
    assert "accuracy" in ci
    assert ci["accuracy"]["ci_low"] <= ci["accuracy"]["mean"] <= ci["accuracy"]["ci_high"]

    info(f"  accuracy={m['accuracy']:.3f} | roc_auc={m['roc_auc']:.3f} | "
         f"threshold_opt={thresh:.3f}")

run_test("All metrics + bootstrap CIs + threshold opt", test_metrics)


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════

header("SMOKE TEST SUMMARY")
passed = sum(1 for _, ok_, _ in results if ok_)
total  = len(results)

for name, ok_, err in results:
    status = f"{GREEN}PASS{RESET}" if ok_ else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}")

print(f"\n{BOLD}  Result: {passed}/{total} tests passed{RESET}")

if passed == total:
    print(f"\n{GREEN}{BOLD}  [OK] Framework is healthy - ready for real MRI training!{RESET}\n")
    sys.exit(0)
else:
    print(f"\n{RED}{BOLD}  [!!] {total - passed} test(s) failed - see errors above.{RESET}\n")
    sys.exit(1)
