# Multi-Site Multi-Sequence MRI Harmonization
This is the official code repository for the paper entitled "Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style" (under review).

## Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style
Mengqi Wu, Yongheng Sun, Qianqian Wang, Pew-Thian Yap, Mingxia Liu

## Abstract
<div align="justify">

Aggregating multi-site brain MRI data can enhance deep learning model training, but also introduces non-biological heterogeneity caused by site-specific variations (e.g., differences in scanner vendors, acquisition parameters, and imaging protocols) that can undermine generalizability. Recent retrospective MRI harmonization seeks to reduce such site effects by standardizing image style (e.g., intensity, contrast, noise patterns) while preserving anatomical content. However, existing methods often rely on limited paired traveling-subject data or fail to effectively disentangle style from anatomy. Furthermore, most current approaches address only single-sequence harmonization, restricting their use in real-world settings where multi-sequence MRI is routinely acquired. To this end, we introduce MMH, **a unified framework for multi-site multi-sequence brain MRI harmonization** that leverages biomedical semantic priors for sequence-aware style alignment. MMH operates in two stages: (1) a diffusion-based global harmonizer that maps MR images to a sequence-specific unified domain using style-agnostic gradient conditioning, and (2) a target-specific fine-tuner that adapts globally aligned images to desired target domains. A tri-planar attention BiomedCLIP encoder aggregates multi-view embeddings to characterize volumetric style information, allowing explicit disentanglement of image styles from anatomy without requiring paired data. Evaluations on 4,163 T1- and T2-weighted MRIs demonstrate MMH’s superior harmonization over state-of-the-art methods in image feature clustering, voxel-level comparison, tissue segmentation, and downstream age and site classification.


## Citation
If you use this code in your research, please cite this paper:

```
@misc{gao2025learning,
  author = {Wu, Mengqi and Sun, Yongheng and Wang, Qianqian and Yap, Pew-Thian and Liu, Mingxia},  
  title = {Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style},  
  archivePrefix={arXiv:2601.08193},
  year = {2026}
}
```
