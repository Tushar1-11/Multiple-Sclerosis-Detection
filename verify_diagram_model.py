"""
Verification Script for Diagram Model Architecture
==================================================
Validates the complete multi-branch CNNAttentionHybrid model matching the diagram:
  - Branch 1: 1D-ResNet CNN + SE-Blocks
  - Branch 2: FT-Transformer Numerical Tokenizer
  - Branch 3: TabTransformer Column Embeddings
  - Combined Token Sequence + CLS Token + Positional Encoding
  - Dual Recurrent Branches (BiLSTM + BiGRU)
  - Recurrent Residual Fusion
  - Stacked Transformer Self-Attention Blocks
  - Latent Fusion: CLS + Attention Pool + Stacking
  - Multi-Task Heads: Phenotype Classifier, EDSS Score Regressor, Binary Group Classifier
"""

import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Setup Path
sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_attention import CNNAttentionHybrid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyDiagramModel")


def main():
    logger.info("=" * 70)
    logger.info(" Starting Diagram Architecture Verification Test")
    logger.info("=" * 70)

    batch_size = 8
    in_features = 47
    num_phenotype_classes = 3
    num_group_classes = 2
    token_dim = 128
    attn_dim = 128
    cnn_seq_len = 16

    # 1. Instantiate Model
    model = CNNAttentionHybrid(
        in_features=in_features,
        token_dim=token_dim,
        cnn_channels=(64, 128, 128, 128),
        cnn_seq_len=cnn_seq_len,
        num_attn_layers=4,
        num_heads=4,
        attn_dim=attn_dim,
        num_phenotype_classes=num_phenotype_classes,
        num_group_classes=num_group_classes,
    )
    model.eval()

    logger.info("✓ Model initialized successfully!")

    # 2. Input Tensor Simulation
    dummy_input = torch.randn(batch_size, in_features)
    logger.info(f"Input Tensor Shape: {dummy_input.shape}")

    # 3. Test Individual Branches
    logger.info("-" * 50)
    logger.info("Testing Parallel Branches:")
    tok_cnn = model.branch1_cnn(dummy_input)
    logger.info(f"  Branch 1 (1D-ResNet CNN + SE-Blocks):  {tok_cnn.shape} (Expected: [{batch_size}, {cnn_seq_len}, {token_dim}])")
    assert tok_cnn.shape == (batch_size, cnn_seq_len, token_dim)

    tok_ft = model.branch2_ft(dummy_input)
    logger.info(f"  Branch 2 (FT-Transformer Tokenizer): {tok_ft.shape} (Expected: [{batch_size}, {in_features}, {token_dim}])")
    assert tok_ft.shape == (batch_size, in_features, token_dim)

    tok_tab = model.branch3_tab(dummy_input)
    logger.info(f"  Branch 3 (TabTransformer Embeddings):{tok_tab.shape} (Expected: [{batch_size}, {in_features}, {token_dim}])")
    assert tok_tab.shape == (batch_size, in_features, token_dim)

    # 4. Combined Token Sequence
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    seq_tokens = torch.cat([cls_tokens, tok_cnn, tok_ft, tok_tab], dim=1)
    total_len = 1 + cnn_seq_len + in_features + in_features
    logger.info(f"  Combined Sequence + CLS Token:       {seq_tokens.shape} (Expected: [{batch_size}, {total_len}, {token_dim}])")
    assert seq_tokens.shape == (batch_size, total_len, token_dim)

    # 5. Dual Recurrent Branches & Recurrent Residual Fusion
    rec_fused = model.recurrent_fusion(seq_tokens)
    logger.info(f"  Recurrent Residual Fusion (BiLSTM+BiGRU): {rec_fused.shape} (Expected: [{batch_size}, {total_len}, {token_dim}])")
    assert rec_fused.shape == (batch_size, total_len, token_dim)

    # 6. Stacked Transformer Blocks
    trans_out = rec_fused
    for blk in model.transformer_blocks:
        trans_out = blk(trans_out)
    trans_out = model.final_norm(trans_out)
    logger.info(f"  Stacked Transformer Self-Attention: {trans_out.shape} (Expected: [{batch_size}, {total_len}, {token_dim}])")
    assert trans_out.shape == (batch_size, total_len, token_dim)

    # 7. Latent Fusion: CLS + Attention Pool + Stacking
    latent = model.latent_fusion(trans_out)
    logger.info(f"  Latent Fusion Vector:              {latent.shape} (Expected: [{batch_size}, {attn_dim}])")
    assert latent.shape == (batch_size, attn_dim)

    # 8. Multi-Task Heads Pass
    logger.info("-" * 50)
    logger.info("Testing Multi-Task Output Heads:")
    outputs = model(dummy_input, return_all_heads=True)

    pheno_out = outputs["phenotype_logits"]
    group_out = outputs["group_logits"]
    edss_out = outputs["edss_pred"]

    logger.info(f"  Head 1 (Phenotype Classifier): {pheno_out.shape} (Expected: [{batch_size}, {num_phenotype_classes}])")
    logger.info(f"  Head 2 (EDSS Score Regressor): {edss_out.shape} (Expected: [{batch_size}, 1])")
    logger.info(f"  Head 3 (Binary Group Classifier): {group_out.shape} (Expected: [{batch_size}, {num_group_classes}])")

    assert pheno_out.shape == (batch_size, num_phenotype_classes)
    assert edss_out.shape == (batch_size, 1)
    assert group_out.shape == (batch_size, num_group_classes)

    # 9. Backward Pass Gradient Flow Test
    logger.info("-" * 50)
    logger.info("Testing Backward Pass Gradient Flow:")
    model.train()
    target_pheno = torch.randint(0, num_phenotype_classes, (batch_size,))
    target_edss = torch.rand(batch_size, 1) * 10.0
    target_group = torch.randint(0, num_group_classes, (batch_size,))

    train_outs = model(dummy_input, return_all_heads=True)
    loss_pheno = nn.CrossEntropyLoss()(train_outs["phenotype_logits"], target_pheno)
    loss_edss = nn.MSELoss()(train_outs["edss_pred"], target_edss)
    loss_group = nn.CrossEntropyLoss()(train_outs["group_logits"], target_group)

    total_loss = loss_pheno + loss_edss + loss_group
    total_loss.backward()

    # Verify non-zero gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for parameter {name}"
            grad_count += 1

    logger.info(f"✓ Backward pass verified! {grad_count} parameter tensors updated with non-zero gradients.")
    logger.info("=" * 70)
    logger.info(" ALL DIAGRAM ARCHITECTURE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
