# LEGO AI Instruction Generation: 3 Project Summaries

This repository includes three different machine learning-based methods for generating or discovering LEGO building instructions:

## 1. LLM Fine-Tuning

**Approach:** Large Language Model (LLM)

This script fine-tunes a pre-trained GPT-2 model using Low-Rank Adaptation (LoRA) on a small corpus of LEGO instructions. It learns to generate fluent, step-by-step text-based building instructions like:

```
Step 1: Attach a 2×4 plate …
Step 2: Stack a 2×2 brick …
```

**Highlights:**
- Uses Hugging Face Transformers and PEFT (LoRA)
- Trains on mock LEGO instruction data
- Generates new instructions from prompts like "Build a spaceship"

---

## 2. Deep Reinforcement Learning

**Approach:** Reinforcement Learning (RL)

This script trains an agent using Proximal Policy Optimization (PPO) to build a target LEGO-like structure inside a 3D grid environment. The agent learns to place blocks correctly, mimicking a building process.

**Highlights:**
- Custom 3D grid environment simulating a Minecraft-like world
- PPO RL agent places blocks to match a target structure
- Action sequence functions like emergent instructions

---

## 3. Graph-Based Deep Learning

**Approach:** Graph Neural Network (GNN)

This script uses a graph-based deep learning model to predict the assembly order of LEGO-like structures. Each brick is a node, and edges represent connections between them. A GCN model outputs per-brick priorities for assembly.

**Highlights:**
- Structures are represented as graphs using PyTorch Geometric
- GCN model regresses to the correct build sequence
- Infers feasible build orders by sorting bricks by predicted scores
