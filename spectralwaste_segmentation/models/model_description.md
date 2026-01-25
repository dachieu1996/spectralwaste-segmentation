# Overview
## MiniNet
```mermaid
graph LR
    subgraph MiniNet["MiniNet Multimodal"]
        M1["RGB + HSI<br/>Concatenate"]
        M2["Shared CNN<br/>Encoder"]
        M3["Decoder"]
        M4["Output"]
        M1 --> M2 --> M3 --> M4
    end

    style MiniNet fill:#e3f2fd
```
## SegFormer
```mermaid
graph LR
    subgraph SegFormer["SegFormer Multimodal"]
        S1["RGB + HSI<br/>Concatenate"]
        S2["Transformer<br/>Encoder"]
        S3["MLP<br/>Decoder"]
        S4["Output"]
        S1 --> S2 --> S3 --> S4
    end

    style SegFormer fill:#f3e5f5
```
## CMX-B0
```mermaid
graph LR
    subgraph CMX["CMX-B0"]
        C1["RGB"]
        C2["HSI"]
        C3["Dual<br/>Encoders"]
        C4["Cross-Modal<br/>Fusion"]
        C5["Decoder"]
        C6["Output"]
        C1 --> C3
        C2 --> C3
        C3 --> C4 --> C5 --> C6
    end

    style CMX fill:#e8f5e9
```
## FuseMoE
```mermaid
graph LR
    subgraph FuseMoE["FuseMoE"]
        F1["RGB + HSI<br/>Concatenate"]
        F2["Conv<br/>Projection"]
        F3["Mixture of<br/>Experts"]
        F4["Gating<br/>Network"]
        F5["Output"]
        F1 --> F2 --> F3
        F3 --> F4
        F4 --> F3
        F3 --> F5
    end

    style FuseMoE fill:#fff3e0
```

## MAE-MoE (PROPOSED)
```mermaid
graph LR
    subgraph MAEMoE["MAE-MoE: Self-Supervised MoE"]
        MM1["RGB + HSI<br/>Separate"]
        MM2["Dual MAE<br/>Pretrain"]
        MM3["MoE Fusion<br/>Per-Modality Router"]
        MM4["Decoder"]
        MM5["Output"]
        MM1 --> MM2 --> MM3 --> MM4 --> MM5
    end

    style MAEMoE fill:#fce4ec
```

## HMoE-Seg (PROPOSED)
```mermaid
graph LR
    subgraph HMoE["HMoE-Seg: Hierarchical MoE"]
        H1["RGB"]
        H2["HSI"]
        H3["Dual<br/>Encoders"]
        H4["Hierarchical<br/>MoE Fusion"]
        H5["Decoder"]
        H6["Output"]
        H1 --> H3
        H2 --> H3
        H3 --> H4 --> H5 --> H6
    end

    style HMoE fill:#e0f2f1
```

## MAE-CMX (PROPOSED)
```mermaid
graph LR
    subgraph MAECMX["MAE-CMX: Pretrained Cross-Modal"]
        MC1["RGB"]
        MC2["HSI"]
        MC3["MAE Pretrained<br/>Encoders"]
        MC4["Cross-Modal<br/>Fusion (FRM+FFM)"]
        MC5["Decoder"]
        MC6["Output"]
        MC1 --> MC3
        MC2 --> MC3
        MC3 --> MC4 --> MC4 --> MC5 --> MC6
    end

    style MAECMX fill:#f1f8e9
```

# Detail

## MiniNet
```mermaid
graph TB
    subgraph MiniNet["MiniNet Multimodal Architecture"]
        direction TB
        MN_IN["RGB + HSI<br/>Concatenated Input"]

        subgraph MN_DOWN["Downsample Block"]
            MN_D1["Downsample 1<br/>→ 16 channels"]
            MN_D2["Downsample 2<br/>→ 64 channels"]
            MN_M1_10["10x MiniNetv2Module<br/>Multi-Dilation DepthwiseSeparable<br/>dilation=1"]
            MN_D3["Downsample 3<br/>→ 128 channels"]
        end

        subgraph MN_FEAT["Feature Extractor Block"]
            MN_M11_26["16x MiniNetv2Module<br/>Multi-Dilation DepthwiseSeparable<br/>dilation=[1,2,1,4,1,8,1,16,...]"]
        end

        subgraph MN_REF["Refinement Block"]
            MN_D4["Downsample 4<br/>→ 16 channels"]
            MN_D5["Downsample 5<br/>→ 64 channels"]
        end

        subgraph MN_UP["Upsample Block"]
            MN_UP1["Upsample 1<br/>128→64 channels"]
            MN_ADD["Add/Skip Connection"]
            MN_M27_30["4x MiniNetv2Module<br/>dilation=1"]
            MN_OUT["Upsample Output<br/>→ num_classes"]
        end

        MN_IN --> MN_D1 --> MN_D2 --> MN_M1_10 --> MN_D3
        MN_D3 --> MN_M11_26
        MN_IN --> MN_D4 --> MN_D5
        MN_M11_26 --> MN_UP1
        MN_UP1 --> MN_ADD
        MN_D5 --> MN_ADD
        MN_ADD --> MN_M27_30 --> MN_OUT
    end

    style MiniNet fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    style MN_DOWN fill:#bbdefb,color:#000
    style MN_FEAT fill:#bbdefb,color:#000
    style MN_REF fill:#bbdefb,color:#000
    style MN_UP fill:#bbdefb,color:#000
```

## SegFormer
```mermaid
graph TB
    subgraph SegFormer["SegFormer Multimodal Architecture"]
        direction TB
        SF_IN["RGB + HSI<br/>Concatenated Input"]

        subgraph SF_ENC["Mix Transformer Encoder (mit_b0)"]
            SF_PE1["OverlapPatchEmbed 1<br/>7x7, stride=4<br/>→ embed_dim[0]"]
            SF_B1["Transformer Blocks 1<br/>depth[0] blocks<br/>Self-Attention + MLP"]
            SF_PE2["OverlapPatchEmbed 2<br/>3x3, stride=2<br/>→ embed_dim[1]"]
            SF_B2["Transformer Blocks 2<br/>depth[1] blocks"]
            SF_PE3["OverlapPatchEmbed 3<br/>3x3, stride=2<br/>→ embed_dim[2]"]
            SF_B3["Transformer Blocks 3<br/>depth[2] blocks"]
            SF_PE4["OverlapPatchEmbed 4<br/>3x3, stride=2<br/>→ embed_dim[3]"]
            SF_B4["Transformer Blocks 4<br/>depth[3] blocks"]
        end

        subgraph SF_DEC["MLP Decoder Head"]
            SF_L1["Linear C1<br/>→ embedding_dim"]
            SF_L2["Linear C2<br/>→ embedding_dim"]
            SF_L3["Linear C3<br/>→ embedding_dim"]
            SF_L4["Linear C4<br/>→ embedding_dim"]
            SF_UP["Upsample All<br/>to C1 size"]
            SF_FUSE["Conv Fuse<br/>4*embedding_dim<br/>→ embedding_dim"]
            SF_PRED["Linear Pred<br/>→ num_classes"]
        end

        SF_IN --> SF_PE1 --> SF_B1 --> SF_PE2 --> SF_B2
        SF_B2 --> SF_PE3 --> SF_B3 --> SF_PE4 --> SF_B4
        SF_B1 --> SF_L1
        SF_B2 --> SF_L2
        SF_B3 --> SF_L3
        SF_B4 --> SF_L4
        SF_L1 & SF_L2 & SF_L3 & SF_L4 --> SF_UP --> SF_FUSE --> SF_PRED
    end

    style SegFormer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    style SF_ENC fill:#e1bee7,color:#000
    style SF_DEC fill:#e1bee7,color:#000
```

## CMX-B0
```mermaid
graph TB
    subgraph CMX["CMX-B0 Dual Encoder Architecture"]
        direction TB
        CMX_RGB["RGB Input"]
        CMX_HSI["HSI Input"]

        subgraph CMX_ENC["Dual Encoders (4 Stages)"]
            direction LR
            subgraph CMX_S1["Stage 1"]
                CMX_PE1_R["PatchEmbed RGB"]
                CMX_PE1_H["PatchEmbed HSI"]
                CMX_B1_R["Transformer Blocks RGB"]
                CMX_B1_H["Transformer Blocks HSI"]
                CMX_FRM1["FRM 1<br/>Feature Rectify<br/>Channel+Spatial Weights"]
                CMX_FFM1["FFM 1<br/>CrossPath + ChannelEmbed"]
            end

            subgraph CMX_S2["Stage 2"]
                CMX_FRM2["FRM 2"]
                CMX_FFM2["FFM 2"]
            end

            subgraph CMX_S3["Stage 3"]
                CMX_FRM3["FRM 3"]
                CMX_FFM3["FFM 3"]
            end

            subgraph CMX_S4["Stage 4"]
                CMX_FRM4["FRM 4"]
                CMX_FFM4["FFM 4"]
            end
        end

        subgraph CMX_DEC["MLP Decoder"]
            CMX_DEC_OUT["Decode Fused Features<br/>→ num_classes"]
        end

        CMX_RGB --> CMX_PE1_R --> CMX_B1_R
        CMX_HSI --> CMX_PE1_H --> CMX_B1_H
        CMX_B1_R & CMX_B1_H --> CMX_FRM1
        CMX_FRM1 --> CMX_FFM1
        CMX_FFM1 --> CMX_FRM2 --> CMX_FFM2
        CMX_FFM2 --> CMX_FRM3 --> CMX_FFM3
        CMX_FFM3 --> CMX_FRM4 --> CMX_FFM4
        CMX_FFM4 --> CMX_DEC_OUT
    end

    style CMX fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style CMX_ENC fill:#c8e6c9,color:#000
    style CMX_S1 fill:#a5d6a7,color:#000
    style CMX_S2 fill:#a5d6a7,color:#000
    style CMX_S3 fill:#a5d6a7,color:#000
    style CMX_S4 fill:#a5d6a7,color:#000
    style CMX_DEC fill:#c8e6c9,color:#000
```

## FuseMoE
```mermaid
graph TB
    subgraph FuseMoE["FuseMoE Sparse MoE Architecture"]
        direction TB
        FM_IN["RGB + HSI<br/>Concatenated Input"]

        subgraph FM_PROJ["Input Projection"]
            FM_CONV["Conv2d 1x1<br/>total_channels → embed_dim"]
            FM_RESHAPE["Reshape to Tokens<br/>[B*H*W, embed_dim]"]
        end

        subgraph FM_MOE["Mixture of Experts Layer"]
            direction TB
            FM_GATE["Gating Network<br/>Noisy Top-K Gating"]
            FM_ROUTER["Router<br/>Select top_k experts<br/>per token"]

            subgraph FM_EXP["Expert Networks (num_experts)"]
                FM_E1["Expert 1<br/>MLP: embed_dim<br/>→ 4*embed_dim<br/>→ num_classes"]
                FM_E2["Expert 2<br/>MLP"]
                FM_E3["..."]
                FM_EN["Expert N<br/>MLP"]
            end

            FM_DISP["SparseDispatcher<br/>Dispatch tokens to experts"]
            FM_COMB["Combine Expert Outputs<br/>Weighted by gates"]
        end

        subgraph FM_OUT["Output"]
            FM_RESHAPE_OUT["Reshape to Image<br/>[B, num_classes, H, W]"]
            FM_AUX["Auxiliary Loss<br/>Load Balancing"]
        end

        FM_IN --> FM_CONV --> FM_RESHAPE
        FM_RESHAPE --> FM_GATE --> FM_ROUTER
        FM_ROUTER --> FM_DISP
        FM_DISP --> FM_E1 & FM_E2 & FM_E3 & FM_EN
        FM_E1 & FM_E2 & FM_E3 & FM_EN --> FM_COMB
        FM_COMB --> FM_RESHAPE_OUT
        FM_ROUTER -.-> FM_AUX
    end

    style FuseMoE fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    style FM_PROJ fill:#ffe0b2,color:#000
    style FM_MOE fill:#ffe0b2,color:#000
    style FM_EXP fill:#ffcc80,color:#000
    style FM_OUT fill:#ffe0b2,color:#000
```

## MAE-MoE (PROPOSED)
```mermaid
graph TB
    subgraph MAEMoE["MAE-MoE: Self-Supervised Pretrained MoE Architecture"]
        direction TB
        MAEM_RGB["RGB Input"]
        MAEM_HSI["HSI Input"]

        subgraph MAEM_PRE["Pretrain Stage (Optional)"]
            direction LR
            subgraph MAEM_MAE_RGB["MAE Encoder RGB"]
                MAEM_PE_R["PatchEmbed Spatial"]
                MAEM_MASK_R["Random Masking<br/>75% mask ratio"]
                MAEM_ENC_R["ViT Encoder<br/>Spatial Branch"]
            end

            subgraph MAEM_MAE_HSI["MAE Encoder HSI"]
                MAEM_PE_H["PatchEmbed Spatial+Channel"]
                MAEM_MASK_H["Random Masking<br/>Dual Branch"]
                MAEM_ENC_H["ViT Encoder<br/>Spatial+Channel Branch"]
            end

            MAEM_DEC["MAE Decoder<br/>Reconstruction"]
            MAEM_LOSS["MSE Loss<br/>on Masked Patches"]
        end

        subgraph MAEM_SEG["Segmentation Stage"]
            direction TB
            MAEM_FEAT_R["Pretrained Features RGB<br/>from MAE Encoder"]
            MAEM_FEAT_H["Pretrained Features HSI<br/>from MAE Encoder"]

            subgraph MAEM_MOE["Per-Modality MoE Fusion"]
                MAEM_GATE_R["Gating Network RGB<br/>Per-Modality Router"]
                MAEM_GATE_H["Gating Network HSI<br/>Per-Modality Router"]
                MAEM_ROUTER["Joint Router<br/>Cross-Modal Selection"]

                subgraph MAEM_EXP["Shared Expert Pool"]
                    MAEM_E1["Expert 1<br/>RGB Specialist"]
                    MAEM_E2["Expert 2<br/>HSI Specialist"]
                    MAEM_E3["Expert 3<br/>Fusion Specialist"]
                    MAEM_EN["Expert N<br/>General"]
                end

                MAEM_DISP["SparseDispatcher<br/>Modality-Aware"]
                MAEM_COMB["Weighted Combination<br/>Load Balancing"]
            end

            MAEM_HEAD["Segmentation Head<br/>→ num_classes"]
        end

        MAEM_RGB --> MAEM_PE_R --> MAEM_MASK_R --> MAEM_ENC_R
        MAEM_HSI --> MAEM_PE_H --> MAEM_MASK_H --> MAEM_ENC_H
        MAEM_ENC_R & MAEM_ENC_H --> MAEM_DEC --> MAEM_LOSS

        MAEM_ENC_R --> MAEM_FEAT_R
        MAEM_ENC_H --> MAEM_FEAT_H
        MAEM_FEAT_R --> MAEM_GATE_R
        MAEM_FEAT_H --> MAEM_GATE_H
        MAEM_GATE_R & MAEM_GATE_H --> MAEM_ROUTER
        MAEM_ROUTER --> MAEM_DISP
        MAEM_DISP --> MAEM_E1 & MAEM_E2 & MAEM_E3 & MAEM_EN
        MAEM_E1 & MAEM_E2 & MAEM_E3 & MAEM_EN --> MAEM_COMB
        MAEM_COMB --> MAEM_HEAD
    end

    style MAEMoE fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    style MAEM_PRE fill:#f8bbd0,color:#000
    style MAEM_MAE_RGB fill:#f48fb1,color:#000
    style MAEM_MAE_HSI fill:#f48fb1,color:#000
    style MAEM_SEG fill:#f8bbd0,color:#000
    style MAEM_MOE fill:#f48fb1,color:#000
    style MAEM_EXP fill:#ec407a,color:#fff
```

## HMoE-Seg (PROPOSED)
```mermaid
graph TB
    subgraph HMoE["HMoE-Seg: Hierarchical Mixture of Experts Segmentation"]
        direction TB
        HMOE_RGB["RGB Input"]
        HMOE_HSI["HSI Input"]

        subgraph HMOE_ENC["Dual Encoders"]
            direction LR
            subgraph HMOE_ENC_R["RGB Encoder"]
                HMOE_S1_R["Stage 1<br/>PatchEmbed + Blocks"]
                HMOE_S2_R["Stage 2<br/>Downsample + Blocks"]
                HMOE_S3_R["Stage 3<br/>Downsample + Blocks"]
                HMOE_S4_R["Stage 4<br/>Downsample + Blocks"]
            end

            subgraph HMOE_ENC_H["HSI Encoder"]
                HMOE_S1_H["Stage 1<br/>PatchEmbed + Blocks"]
                HMOE_S2_H["Stage 2<br/>Downsample + Blocks"]
                HMOE_S3_H["Stage 3<br/>Downsample + Blocks"]
                HMOE_S4_H["Stage 4<br/>Downsample + Blocks"]
            end
        end

        subgraph HMOE_FUSION["Hierarchical MoE Fusion"]
            direction TB

            subgraph HMOE_L1["Level 1: Outer MoE (Coarse)"]
                HMOE_GATE_O["Outer Gating<br/>Modality-Level Selection"]
                HMOE_EXP_O1["Outer Expert 1<br/>RGB-Dominant"]
                HMOE_EXP_O2["Outer Expert 2<br/>HSI-Dominant"]
                HMOE_EXP_O3["Outer Expert 3<br/>Balanced"]
            end

            subgraph HMOE_L2["Level 2: Inner MoE (Fine)"]
                HMOE_GATE_I["Inner Gating<br/>Feature-Level Selection"]
                HMOE_EXP_I1["Inner Expert 1<br/>Low-Level Features"]
                HMOE_EXP_I2["Inner Expert 2<br/>Mid-Level Features"]
                HMOE_EXP_I3["Inner Expert 3<br/>High-Level Features"]
                HMOE_EXP_I4["Inner Expert 4<br/>Semantic Features"]
            end

            HMOE_COMB["Hierarchical Combination<br/>Weighted Aggregation"]
        end

        subgraph HMOE_DEC["Decoder"]
            HMOE_UP1["Upsample 1<br/>+ Skip Connection"]
            HMOE_UP2["Upsample 2<br/>+ Skip Connection"]
            HMOE_UP3["Upsample 3<br/>+ Skip Connection"]
            HMOE_HEAD["Segmentation Head<br/>→ num_classes"]
        end

        HMOE_RGB --> HMOE_S1_R --> HMOE_S2_R --> HMOE_S3_R --> HMOE_S4_R
        HMOE_HSI --> HMOE_S1_H --> HMOE_S2_H --> HMOE_S3_H --> HMOE_S4_H

        HMOE_S4_R & HMOE_S4_H --> HMOE_GATE_O
        HMOE_GATE_O --> HMOE_EXP_O1 & HMOE_EXP_O2 & HMOE_EXP_O3
        HMOE_EXP_O1 & HMOE_EXP_O2 & HMOE_EXP_O3 --> HMOE_GATE_I
        HMOE_GATE_I --> HMOE_EXP_I1 & HMOE_EXP_I2 & HMOE_EXP_I3 & HMOE_EXP_I4
        HMOE_EXP_I1 & HMOE_EXP_I2 & HMOE_EXP_I3 & HMOE_EXP_I4 --> HMOE_COMB

        HMOE_COMB --> HMOE_UP1
        HMOE_S3_R & HMOE_S3_H --> HMOE_UP1
        HMOE_UP1 --> HMOE_UP2
        HMOE_S2_R & HMOE_S2_H --> HMOE_UP2
        HMOE_UP2 --> HMOE_UP3
        HMOE_S1_R & HMOE_S1_H --> HMOE_UP3
        HMOE_UP3 --> HMOE_HEAD
    end

    style HMoE fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#000
    style HMOE_ENC fill:#b2dfdb,color:#000
    style HMOE_ENC_R fill:#80cbc4,color:#000
    style HMOE_ENC_H fill:#80cbc4,color:#000
    style HMOE_FUSION fill:#b2dfdb,color:#000
    style HMOE_L1 fill:#4db6ac,color:#000
    style HMOE_L2 fill:#4db6ac,color:#000
    style HMOE_DEC fill:#b2dfdb,color:#000
```

## MAE-CMX (PROPOSED)
```mermaid
graph TB
    subgraph MAECMX["MAE-CMX: MAE Pretrained Cross-Modal Transformer"]
        direction TB
        MCMX_RGB["RGB Input"]
        MCMX_HSI["HSI Input"]

        subgraph MCMX_PRE["Pretrain Stage (Optional)"]
            direction LR
            subgraph MCMX_MAE_R["MAE RGB"]
                MCMX_MASK_R["Masking 75%"]
                MCMX_ENC_R["ViT Encoder"]
                MCMX_DEC_R["Decoder"]
            end

            subgraph MCMX_MAE_H["MAE HSI"]
                MCMX_MASK_H["Masking 75%<br/>Spatial+Channel"]
                MCMX_ENC_H["ViT Encoder<br/>Dual Branch"]
                MCMX_DEC_H["Decoder"]
            end

            MCMX_LOSS_P["Reconstruction Loss"]
        end

        subgraph MCMX_ENC["Dual Pretrained Encoders (4 Stages)"]
            direction LR
            subgraph MCMX_S1["Stage 1"]
                MCMX_PE1_R["PatchEmbed RGB<br/>Initialized from MAE"]
                MCMX_PE1_H["PatchEmbed HSI<br/>Initialized from MAE"]
                MCMX_B1_R["Transformer Blocks RGB<br/>Pretrained Weights"]
                MCMX_B1_H["Transformer Blocks HSI<br/>Pretrained Weights"]
                MCMX_FRM1["FRM 1<br/>Feature Rectify Module<br/>Channel+Spatial Attention"]
                MCMX_FFM1["FFM 1<br/>Feature Fusion Module<br/>CrossPath + ChannelEmbed"]
            end

            subgraph MCMX_S2["Stage 2"]
                MCMX_B2_R["Blocks RGB"]
                MCMX_B2_H["Blocks HSI"]
                MCMX_FRM2["FRM 2"]
                MCMX_FFM2["FFM 2"]
            end

            subgraph MCMX_S3["Stage 3"]
                MCMX_B3_R["Blocks RGB"]
                MCMX_B3_H["Blocks HSI"]
                MCMX_FRM3["FRM 3"]
                MCMX_FFM3["FFM 3"]
            end

            subgraph MCMX_S4["Stage 4"]
                MCMX_B4_R["Blocks RGB"]
                MCMX_B4_H["Blocks HSI"]
                MCMX_FRM4["FRM 4"]
                MCMX_FFM4["FFM 4"]
            end
        end

        subgraph MCMX_DEC["MLP Decoder"]
            MCMX_L1["Linear C1<br/>→ embed_dim"]
            MCMX_L2["Linear C2<br/>→ embed_dim"]
            MCMX_L3["Linear C3<br/>→ embed_dim"]
            MCMX_L4["Linear C4<br/>→ embed_dim"]
            MCMX_UP["Upsample All<br/>to C1 size"]
            MCMX_FUSE["Conv Fuse<br/>→ embed_dim"]
            MCMX_PRED["Linear Pred<br/>→ num_classes"]
        end

        MCMX_RGB --> MCMX_MASK_R --> MCMX_ENC_R --> MCMX_DEC_R
        MCMX_HSI --> MCMX_MASK_H --> MCMX_ENC_H --> MCMX_DEC_H
        MCMX_DEC_R & MCMX_DEC_H --> MCMX_LOSS_P

        MCMX_RGB --> MCMX_PE1_R --> MCMX_B1_R
        MCMX_HSI --> MCMX_PE1_H --> MCMX_B1_H
        MCMX_B1_R & MCMX_B1_H --> MCMX_FRM1 --> MCMX_FFM1

        MCMX_FFM1 --> MCMX_B2_R & MCMX_B2_H
        MCMX_B2_R & MCMX_B2_H --> MCMX_FRM2 --> MCMX_FFM2

        MCMX_FFM2 --> MCMX_B3_R & MCMX_B3_H
        MCMX_B3_R & MCMX_B3_H --> MCMX_FRM3 --> MCMX_FFM3

        MCMX_FFM3 --> MCMX_B4_R & MCMX_B4_H
        MCMX_B4_R & MCMX_B4_H --> MCMX_FRM4 --> MCMX_FFM4

        MCMX_FFM1 --> MCMX_L1
        MCMX_FFM2 --> MCMX_L2
        MCMX_FFM3 --> MCMX_L3
        MCMX_FFM4 --> MCMX_L4

        MCMX_L1 & MCMX_L2 & MCMX_L3 & MCMX_L4 --> MCMX_UP --> MCMX_FUSE --> MCMX_PRED
    end

    style MAECMX fill:#f1f8e9,stroke:#689f38,stroke-width:2px,color:#000
    style MCMX_PRE fill:#dcedc8,color:#000
    style MCMX_MAE_R fill:#c5e1a5,color:#000
    style MCMX_MAE_H fill:#c5e1a5,color:#000
    style MCMX_ENC fill:#dcedc8,color:#000
    style MCMX_S1 fill:#aed581,color:#000
    style MCMX_S2 fill:#aed581,color:#000
    style MCMX_S3 fill:#aed581,color:#000
    style MCMX_S4 fill:#aed581,color:#000
    style MCMX_DEC fill:#dcedc8,color:#000
```

# Proposed Models Summary

## 1. MAE-MoE: Self-Supervised Pretrained Mixture of Experts
**Key Innovation**: Combines self-supervised pretraining (MAE) with adaptive expert routing (MoE)

**Advantages**:
- **Better initialization**: MAE pretraining learns robust representations from unlabeled RGB+HSI data
- **Modality-specific experts**: Per-modality router allows specialized experts for RGB vs HSI
- **Efficient inference**: Sparse activation (top-k experts) reduces computation
- **Transfer learning**: Pretrained encoder can be reused across different waste segmentation tasks

**Architecture Details**:
- Pretrain stage: Dual MAE encoders (RGB spatial + HSI spatial+channel) with 75% masking
- Segmentation stage: Per-modality MoE with joint router for cross-modal expert selection
- Expert specialization: RGB-specialist, HSI-specialist, fusion-specialist, and general experts
- Load balancing loss ensures all experts are utilized

**Best for**: Scenarios with limited labeled data but abundant unlabeled RGB+HSI pairs

---

## 2. HMoE-Seg: Hierarchical Mixture of Experts Segmentation
**Key Innovation**: Two-level hierarchical MoE for coarse-to-fine multimodal fusion

**Advantages**:
- **Hierarchical fusion**: Outer MoE selects modality-level strategy, inner MoE refines feature-level fusion
- **Multi-scale features**: Skip connections from all encoder stages to decoder
- **Adaptive fusion**: Different fusion strategies at different semantic levels
- **Interpretability**: Outer experts show which modality dominates for different regions

**Architecture Details**:
- Dual encoders: Separate RGB and HSI encoders with 4 stages each
- Outer MoE (Level 1): 3 experts for RGB-dominant, HSI-dominant, and balanced fusion
- Inner MoE (Level 2): 4 experts for low/mid/high/semantic level features
- Decoder: Progressive upsampling with skip connections from all stages

**Best for**: Complex scenes where different regions require different fusion strategies (e.g., some waste types more visible in RGB, others in HSI)

---

## 3. MAE-CMX: MAE Pretrained Cross-Modal Transformer
**Key Innovation**: Enhances CMX with MAE pretraining for better initialization

**Advantages**:
- **Strong baseline**: Builds on proven CMX architecture (FRM+FFM modules)
- **Better convergence**: MAE pretrained weights provide better starting point
- **Dual-branch MAE**: Spatial+channel masking for HSI captures spectral correlations
- **Cross-modal attention**: FRM and FFM modules explicitly model RGB-HSI interactions

**Architecture Details**:
- Pretrain: Separate MAE for RGB (spatial only) and HSI (spatial+channel dual branch)
- Encoder: 4-stage dual encoders initialized from MAE weights
- Fusion: FRM (Feature Rectify Module) + FFM (Feature Fusion Module) at each stage
- Decoder: MLP decoder with multi-scale feature aggregation

**Best for**: When you have unlabeled data for pretraining and want to leverage CMX's proven cross-modal fusion

---

## Comparison Table

| Model | Pretraining | Fusion Strategy | Complexity | Best Use Case |
|-------|-------------|-----------------|------------|---------------|
| **FuseMoE** | ❌ | Sparse MoE (joint) | Medium | Baseline MoE approach |
| **MAE-MoE** | ✅ MAE | Per-modality MoE | High | Limited labeled data |
| **HMoE-Seg** | ❌ | Hierarchical MoE | Very High | Complex scenes, interpretability |
| **MAE-CMX** | ✅ MAE | FRM+FFM (CMX) | High | Leverage CMX + pretraining |

## Implementation Priority

1. **MAE-MoE** - Most innovative, combines best of both base models
2. **MAE-CMX** - Easier to implement, builds on existing CMX
3. **HMoE-Seg** - Most complex, implement if others show MoE benefits