# VersaPSE: Versatile Password Strength Evaluation Using Continual Learning

Authors: Yifei Zhang, Zhenduo Hou, Ding Wang

The code presented here is mainly authored by Yifei Zhang (yifeizhang@mail.nankai.edu.cn, GitHub: FeliceRivarez). If you have any questions regarding the code in this repository, please contact Yifei Zhang.

## Introduction

VersaPSE is a password strength evaluation framework that can be instantiated using *any* deep-learning-based credential tweaking model. It features:

1. **Versatile password strength evaluation.** Under the VersaPSE framework, credential tweaking models can be transformed into multipurpose PSMs via continual learning, which can evaluate password strength against *both* credential tweaking and trawling guessing attacks.

2. **Low latency, high deployability.** VersaPSE leverages the probabilistic output of credential tweaking models, as it is a natural representation of password strength. This approach allows multipurpose PSMs derived using VersaPSE to be up to *90 times faster* than reuse-based PSMs that requires the generation of more than 1,000 password guesses on the fly. VersaPSE can also be deployed in browsers, *achieving sub-second latency using CPU-only inference.*

3. **High scability.** VersaPSE is a model-independent framework that can be instantiated using *any* deep-learning-based credential tweaking models. It can be readily applied to any future credential tweaking models, enabling them to enhance password security and protect users' privacy.

## Ethical considerations

**Almost all datasets** involved in our paper contain sensitive information about real-world users (e.g., passwords, PII fields). Therefore, we do not include these datasets in this artifact. However, we do provide some made-up passwords for the sake of reproducing the full workflow of VersaPSE.

Since we utilize various credential tweaking models for PSM construction, we provide the corresponding models as instantiations of VersaPSE. However, to prevent abuse of these models:

1. For models that we have reproduced (e.g., Pass2Edit because it has not been open-sourced, Pass2Path which we re-implemented in PyTorch), we only provide our implementation **without** password-guessing-related code snippets or functionalities.

2. For models that have been made publicly available (KNNGuess), we use their source code **without** providing corresponding model weights.

## Contents

1. KNNGuess without model checkpoints

2. Pass2Edit and PointerGuess (reproduced), with model checkpoints

3. PassBERT and Pass2Path (re-implemented in PyTorch), with model checkpoints

4. Data preparation and misc. code snippets

5. Pseudo password datasets for workflow verification

Additional README files can be found in each folder, as different credential tweaking models vary in terms of design, structure, and functionality.

As for the browser implementation, you can find the demo in Pass2Edit-instantiated VersaPSE. If you want to deploy other instantiations of VersaPSE in a browser, just follow the workflow in Pass2Edit.

## Environment

See requirements.txt, and the README_orignal.md for KNNGuess.

If you run into problems downloading the packages, see requirements_raw.txt

## Acknowledgment

Referenced code and implementations are acknowledged in the README.md of each subfolder.

