# Fuzzy-Embedded Attention-SAC Microgrid EMS

This repository contains the refactored, high-performance codebase for the **Fuzzy-Embedded Safety-Constrained Hierarchical SAC Energy Management System (EMS)** applied to a Floating PV-Battery-Diesel Microgrid. 

The project introduces a novel **Attention-based Recurrent Soft Actor-Critic (SAC)** algorithm combined with a **Supervisory Fuzzy Logic** layer. This layer ensures strict constraint adherence (e.g., SOC boundaries, physical battery limits) while actively guiding the reinforcement learning agent for stable, physically sound dispatch commands.

## Key Features
- **Attention-based Temporal Encoding:** Leverages a Transformer encoder layer over a historical window of states to intuitively handle partial observability and weather/load forecast trends.
- **Fuzzy-Supervisory Constraints:** A strict, unbreakable 4-tier safety projection layer based on Fuzzy Logic protects microgrid assets.
- **Modern DL Practices:** Built with Gradient Clipping, Cosine Annealing Learning Rate Schedulers, and best-model checkpointing tracking using 20-episode moving averages.
- **Fully Modular Architecture:** The 1,900-line monolithic script has been abstracted into standard, clean, object-oriented Python modules (`src/`), making it simple to manage and modify.

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start
You can run the default training sequence immediately. By default, it will execute on CPU for 800 episodes:
```bash
python main.py
```

## Advanced Usage
The entry point `main.py` utilizes `argparse` to let you easily adjust configurations, which is especially useful when transitioning from a local CPU machine to Google Colab.

**Run on Colab (GPU) for 1500 episodes with an attention history window of 8:**
```bash
python main.py --device cuda --episodes 1500 --seq_len 8
```

**Run a quick debug test:**
```bash
python main.py --episodes 5 --warmup_steps 100 --batch_size 64
```

### Full Argument List
- `--train_file`: Path to training CSV (default: `data/load_with_weather.csv`)
- `--test_file`: Path to testing CSV (default: `data/target_load.csv`)
- `--episodes`: Total training episodes (default: 800)
- `--batch_size`: Replay buffer batch size (default: 256)
- `--seq_len`: Sequence length for the Attention Encoder window (default: 4)
- `--warmup_steps`: Number of steps to take random actions before utilizing the policy (default: 3000)
- `--updates_per_step`: Gradient updates per environment step (default: 1)
- `--seed`: Random seed (default: 42)
- `--device`: Target device, e.g., `cpu` or `cuda` (default: `cpu`)

## Outputs and Evaluation
During training, a live progress bar will track the latest reward, the 20-episode Moving Average (MA20), and the best reward achieved so far. 

Once training completes, the system automatically loads the **best performing model** (highest MA20) and evaluates it on the test dataset.

All results are saved automatically into a timestamped directory (e.g., `outputs/`):
- **`best_model/`**: The `.pth` files containing the optimal policy and Q-networks.
- **`final_model_.../`**: Checkpoint representing the very last episode of training.
- **`plots_.../`**: Publication-ready charts that recreate the manuscript figures (Power profiles, SOC curves, Fuzzy shaping tracking, and loss/reward learning curves).
- **`results_...csv`**: Microsecond-level operational logging for the entire testing horizon.
- **`kpis_...csv`**: Final Key Performance Indicators (LPSP, Fuel usage, Degradation cost, etc.).
