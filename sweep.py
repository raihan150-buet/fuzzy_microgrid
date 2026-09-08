import numpy as np
import pandas as pd
from src.utils import set_seed, load_data
from src.env import FuzzyEmbeddedSACMicrogridEnv
from src.sac import SACAgent
from src.buffer import ReplayBuffer

def run_experiment(fuzzy_gain, updates, episodes=40):
    set_seed(42)
    df_train = load_data("data/load_with_weather.csv", start_date="2016-01-01", end_date="2016-12-31 23:59:59")
    
    env = FuzzyEmbeddedSACMicrogridEnv(
        df_train, dt_h=0.25, episode_len=96, forecast_horizon=4, seq_len=4, seed=42, fuzzy_action_gain=fuzzy_gain
    )
    
    agent = SACAgent(
        state_dim=env.state_dim, action_dim=env.action_dim, seq_len=4, device="cpu", max_episodes=episodes
    )
    replay = ReplayBuffer(capacity=50000)
    
    rewards = []
    total_steps = 0
    warmup = 1000
    
    for ep in range(episodes):
        state = env.reset(random_start=True)
        done = False
        ep_reward = 0.0
        
        while not done:
            if total_steps < warmup:
                action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, evaluate=False)
                
            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            
            if total_steps >= warmup:
                for _ in range(updates):
                    agent.train_step(replay, batch_size=256)
                    
            state = next_state
            ep_reward += reward
            total_steps += 1
            
        rewards.append(ep_reward)
        if total_steps >= warmup:
            agent.step_schedulers(np.mean(rewards[-10:]))
            
    final_ma = np.mean(rewards[-10:])
    return final_ma

if __name__ == "__main__":
    results = []
    for gain in [0.0, 0.15, 0.35, 0.60]:
        for upd in [1, 3]:
            print(f"Testing Gain={gain}, Updates={upd}...")
            ma = run_experiment(gain, upd)
            print(f"Result -> Gain: {gain}, Updates: {upd}, Final MA10: {ma:.2f}")
            results.append({"gain": gain, "updates": upd, "ma10": ma})
            
    df = pd.DataFrame(results)
    df.to_csv("sweep_results.csv", index=False)
    print("\nBest configs based on 40-episode early convergence:")
    print(df.sort_values(by="ma10", ascending=False))
