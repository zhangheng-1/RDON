# A Reversible Deep Operator Network for Grid-Independent, Multi-Scale Magnetotelluric Inversion and Uncertainty Quantification

## Key Points
- We propose a novel network architecture RDON that can directly generalize to arbitrarily sparse and irregular station–frequency observations without retraining.
- RDON trained on a single scale can generalize zero-shot to unseen scales with transfer learning correcting scale-induced biases.
- A Bootstrap Resampling scheme  based on RDON enables efficient uncertainty quantification of solutions.

## Abstract
This study proposes a novel deep learning framework, the Reversible Deep Operator Network (RDON), for efficient inversion of arbitrarily sparse magnetotelluric (MT) data. Traditional deep learning methods struggle with sparse and irregularly gridded data, requiring retraining for each new observation setup. RDON leverages a RealNVP-based invertible neural network integrated into a DeepONet architecture to establish a bijective mapping between subsurface resistivity and MT data, enabling grid-independent forward and inverse modeling. The framework demonstrates robust generalization across varying station configurations, frequencies and spatial scales, achieving high-precision inversion without retraining. As to scaling generalization, we establish the Scaling Invariance theorem for MT data, providing both theoretical grounding and rigorous quantification of cross-scale generalization errors. We incorporate a Bootstrap Resampling-based approach for uncertainty quantification on RDON to get significant efficiency improvements over conventional methods. Applied to real-world data from the West Junggar region, RDON validates its reliability and practicality.

## Key Features
- 🧠 **Bijective Operator Learning**: Reversible architecture enables dual-directional modeling
- 🌐 **Grid-Independent inversion**: Handles arbitrary station/frequency configurations
- 📐 **Scale-Invariant Predictions**: Theoretical guarantees for cross-scale generalization
- 📊 **Efficient Uncertainty Quantification**: Bootstrap resampling for solution confidence
- ⚡ **Real data test**: Real-time inversion capabilities

