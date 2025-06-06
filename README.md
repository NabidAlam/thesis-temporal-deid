# MaskAnyone-Temporal: Toolkit for Privacy-Preserving Video De-identification

This repository contains the official codebase for my Master's thesis at the University of Potsdam. The project builds on the [MaskAnyone](https://github.com/MaskAnyone/MaskAnyone) framework and integrates temporal segmentation techniques via [TSP-SAM](https://github.com/WenjunHui1/TSP-SAM) and [SAMURAI](https://github.com/yangchris11/samurai).

The goal: Enable **temporally consistent video de-identification** while preserving key behavioral cues like body pose and gestures, crucial for research in social interaction, medicine, sports and beyond.

---

## Key Features

- Frame-by-frame de-identification via MaskAnyone
- Temporal segmentation using:
  - **TSP-SAM** (frequency-based motion cues)
  - **SAMURAI** (Kalman filter memory tracking)
- Designed for behavioral video analytics
- Modular structure — easy to experiment or extend

---
