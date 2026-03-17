# BirdNET Model Converter - Terms of Use

This document governs use of the BirdNET Model Converter software and trained model weights (the "Model") distributed with or converted by it. By using the Model, you agree to these terms.

## 1. Description
BirdNET Model Converter provides scripts and tools to convert BirdNET V3.0 PyTorch audio classification models into TensorFlow/LiteRT, TorchScript, and ONNX formats. The converted models retain the original model's capabilities for bird species identification from audio.

Development and support involve the K. Lisa Yang Center for Conservation Bioacoustics (Cornell University), Chemnitz University of Technology and Museum für Naturkunde Berlin.

## 2. License
The source code is licensed under the MIT License (see [LICENSE](LICENSE)).

Trained model weights are licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0):
- You may use (including commercially), share, and adapt with proper credit, license link, and indication of changes.
- Derivatives must use the same license.

The prohibitions in Section 5 override permissive license aspects.

## 3. No Warranty
THE MODEL AND SOFTWARE ARE PROVIDED "AS IS" WITHOUT ANY WARRANTY (INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY, OR NON-INFRINGEMENT). USE AT YOUR OWN RISK.

## 4. Data & Responsibility
You are responsible for:
- Understanding that model outputs are probabilistic predictions, not confirmed species observations (false positives and false negatives are possible).
- Verifying that converted models produce outputs consistent with the original PyTorch model.
- Appropriate interpretation of species identification results for your use case.
- Compliance with all applicable regulations regarding species data and conservation.

## 5. Prohibited Uses
You must NOT use the Model or any derivative:
1. For POACHING, wildlife tracking intended to facilitate illegal capture, harm, trade, or exploitation of species.
2. For ANY MILITARY purpose, including surveillance, targeting, or integration into defense systems.

If you cannot comply, do not use the Model.

## 6. Attribution
Publications, presentations, or derived tools must attribute the use of the Model either through citation or acknowledgment (e.g., "Powered by BirdNET").

## 7. Limitation of Liability
Contributors and affiliated institutions are not liable for indirect, incidental, consequential, or special damages arising from use or inability to use the Model.

## 8. Contact
For permissions, clarifications, or to report misuse: stefan.kahl@cornell.edu
