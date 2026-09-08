import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils import set_seed, load_data, compute_kpis
from src.env import FuzzyEmbeddedSACMicrogridEnv
from src.sac import SACAgent
from src.buffer import ReplayBuffer
from src.plotting import plot_results

def evaluate_policy_reward(env, agent):
    state = env.reset(random_start=False)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return float(total_reward)

def evaluate_agent(env, agent):
    state = env.reset(random_start=False)
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)
    return env.get_results()

def main():
    parser = argparse.ArgumentParser(description="Fuzzy-Embedded Attention-SAC Microgrid EMS")
    parser.add_argument("--train_file", type=str, default="data/load_with_weather.csv")
    parser.add_argument("--test_file", type=str, default="data/target_load.csv")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=4, help="Sequence length for Attention Encoder (Novelty)")
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading datasets...")
    df_train = load_data(args.train_file, start_date="2016-01-01", end_date="2018-12-31 23:59:59")
    df_test = load_data(args.test_file)

    dt_h = 15 / 60

    print("Initializing Environments...")
    train_env = FuzzyEmbeddedSACMicrogridEnv(
        df_train, dt_h=dt_h, episode_len=96, forecast_horizon=4, seq_len=args.seq_len, seed=args.seed
    )
    
    test_env = FuzzyEmbeddedSACMicrogridEnv(
        df_test, dt_h=dt_h, episode_len=max(len(df_test) - 5, 1), forecast_horizon=4, seq_len=args.seq_len, seed=99
    )

    agent = SACAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        seq_len=args.seq_len,
        device=args.device,
        max_episodes=args.episodes
    )
    
    replay = ReplayBuffer(capacity=500000)

    rewards = []
    moving_rewards = []
    validation_rewards = []
    loss_history = {"q1_loss": [], "q2_loss": [], "policy_loss": [], "alpha_loss": [], "alpha": []}

    total_steps = 0

    from tqdm import tqdm
    print(f"Starting Training for {args.episodes} episodes...")
    best_reward = -float('inf')
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model")

    pbar = tqdm(range(args.episodes), desc="Training")
    for ep in pbar:
        state = train_env.reset(random_start=True)
        done = False
        ep_reward = 0.0

        while not done:
            if total_steps < args.warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=train_env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, _ = train_env.step(action)
            replay.push(state, action, reward, next_state, float(done))

            if total_steps >= args.warmup_steps:
                for _ in range(args.updates_per_step):
                    info = agent.train_step(replay, batch_size=args.batch_size)
                    if info is not None:
                        for k in loss_history:
                            loss_history[k].append(info[k])

            state = next_state
            ep_reward += reward
            total_steps += 1

        rewards.append(ep_reward)
        current_ma = float(pd.Series(rewards).rolling(20, min_periods=1).mean().iloc[-1])
        moving_rewards.append(current_ma)
        
        # Only step schedulers if training has actually started (optimizers have stepped)
        if total_steps >= args.warmup_steps:
            agent.step_schedulers()

        # Update progress bar
        pbar.set_postfix({
            "Reward": f"{ep_reward:.2f}",
            "MA20": f"{current_ma:.2f}",
            "Best": f"{best_reward:.2f}" if best_reward != -float('inf') else "N/A"
        })

        # Save best model based on 20-episode moving average
        if current_ma > best_reward and ep >= 20:
            best_reward = current_ma
            agent.save(best_model_path)

        if (ep + 1) % 20 == 0:
            val_reward = evaluate_policy_reward(train_env, agent)
            validation_rewards.append({"episode": ep + 1, "validation_reward": val_reward})

    print(f"Loading best model (MA20 Reward: {best_reward:.2f}) for testing...")
    # The agent doesn't have a load method, but we saved it. To test with it we should ideally load it.
    # We will implement a quick load below.
    try:
        agent.policy.load_state_dict(torch.load(os.path.join(best_model_path, "policy.pth")))
        agent.q1.load_state_dict(torch.load(os.path.join(best_model_path, "q1.pth")))
        agent.q2.load_state_dict(torch.load(os.path.join(best_model_path, "q2.pth")))
    except Exception as e:
        print(f"Could not load best model, using final model. Error: {e}")

    # Evaluation
    print("Evaluating on test set...")
    results = evaluate_agent(test_env, agent)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    agent.save(os.path.join(output_dir, f"final_model_{timestamp_tag}"))

    results_file = os.path.join(output_dir, f"results_{timestamp_tag}.csv")
    kpi_file = os.path.join(output_dir, f"kpis_{timestamp_tag}.csv")
    plot_dir = os.path.join(output_dir, f"plots_{timestamp_tag}")

    results.to_csv(results_file)
    kpis = compute_kpis(results, dt_h=dt_h, fuel_cost_per_l=1.0)
    kpis.to_csv(kpi_file, index=False)

    print("\n===== KPI SUMMARY ON TEST DATA =====")
    print(kpis.T)

    plot_results(results, rewards, moving_rewards, validation_rewards, loss_history, save_dir=plot_dir)
    print(f"\nTraining complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
