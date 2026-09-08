import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(
    results,
    rewards,
    moving_rewards=None,
    validation_rewards=None,
    loss_history=None,
    save_dir="plots"
):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label="Episode Reward")
    if moving_rewards is not None:
        plt.plot(moving_rewards, label="20-Episode Moving Average")
    if validation_rewards:
        val_df = pd.DataFrame(validation_rewards)
        if len(val_df) > 0:
            plt.plot(val_df["episode"], val_df["validation_reward"], marker="o", label="Validation Reward")
    plt.xlabel("Episode")
    plt.ylabel("Scaled Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_reward.png", dpi=300)
    plt.close()

    if loss_history is not None:
        for key, ylabel in [("q1_loss", "Q1 Loss"), ("q2_loss", "Q2 Loss"), ("policy_loss", "Policy Loss"), ("alpha", "Entropy Temperature Alpha")]:
            values = loss_history.get(key, [])
            if len(values) > 0:
                plt.figure(figsize=(12, 5))
                plt.plot(pd.Series(values).rolling(300, min_periods=1).mean(), label=f"{ylabel} MA300")
                plt.xlabel("Training Update")
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{key}_ma.png", dpi=300)
                plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["load_kw"], "k--", label="Load")
    plt.plot(results.index, results["pv_ac_kw"], label="PV Available")
    plt.plot(results.index, results["pv_after_curt_kw"], label="PV After Curtailment")
    plt.plot(results.index, results["batt_discharge_kw_positive"], label="Battery Discharge")
    plt.plot(results.index, -results["batt_charge_kw_positive"], label="Battery Charge")
    plt.plot(results.index, results["diesel_kw"], label="Diesel")
    plt.ylabel("Power (kW)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/power_profile.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["p_batt_ref_raw_kw"], label="Raw SAC Battery Ref")
    plt.plot(results.index, results["p_batt_ref_fuzzy_kw"], label="Fuzzy-Shaped Battery Ref")
    plt.plot(results.index, results["p_batt_cmd_safe_kw"], label="Safe Battery Command")
    plt.plot(results.index, results["batt_ac_kw"], label="Actual Battery")
    plt.ylabel("Battery Power (kW)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/battery_commands.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["soc"], label="SOC")
    plt.plot(results.index, results["raw_soc_target"], label="Raw SAC SOC Target")
    plt.plot(results.index, results["fuzzy_soc_target"], label="Fuzzy-Shaped SOC Target")
    plt.ylabel("SOC")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/soc_target.png", dpi=300)
    plt.close()
    
    print(f"Plots saved in: {save_dir}")
