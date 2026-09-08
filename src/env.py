import numpy as np
import pandas as pd
from collections import deque
from src.components import FloatingPVArray, DegradationBattery, BatteryConverter, DynamicDieselGen, LVLineAdvanced
from src.fuzzy_logic import FuzzyEmbeddedSupervisor

class FuzzyEmbeddedSACMicrogridEnv:
    """
    Microgrid environment utilizing fuzzy logic for supervisory control.
    Supports temporal state sequences if seq_len > 1.
    """
    def __init__(
        self,
        df,
        dt_h=0.25,
        episode_len=96,
        forecast_horizon=4,
        seq_len=1,
        seed=42,
        reward_scale=100.0,
        reward_clip=(-50.0, 10.0),
        fuzzy_action_gain=0.35,
        fuzzy_penalty_weight=20.0
    ):
        self.df = df.reset_index(drop=True)
        self.dt_h = float(dt_h)
        self.dt_s = self.dt_h * 3600.0
        self.episode_len = int(episode_len)
        self.forecast_horizon = int(forecast_horizon)
        self.seq_len = int(seq_len)
        self.rng = np.random.default_rng(seed)
        self.reward_scale = float(reward_scale)
        self.reward_clip = reward_clip
        self.fuzzy_action_gain = float(fuzzy_action_gain)
        self.fuzzy_penalty_weight = float(fuzzy_penalty_weight)

        self.pv = FloatingPVArray(p_stc_kw=250.0)
        self.battery = DegradationBattery(e_capacity_kwh=1000.0, soc_init=0.60)
        self.converter = BatteryConverter()
        self.diesel = DynamicDieselGen()
        self.line = LVLineAdvanced()

        self.fuzzy = FuzzyEmbeddedSupervisor(
            soc_min=self.battery.soc_min,
            soc_max=self.battery.soc_max
        )

        self.max_load = max(float(self.df["load_kw"].max()), 1e-6)
        self.max_pv = self.pv.p_stc_kw
        self.norm_power = max(self.max_load, self.max_pv, self.diesel.p_rated_kw, 1.0)

        self.idx = 0
        self.steps = 0
        self.records = []
        self.state_buffer = deque(maxlen=self.seq_len)

        self.prev_batt_ac_kw = 0.0
        self.prev_diesel_kw = 0.0
        self.prev_dg_on = 0.0

    @property
    def state_dim(self):
        return 14

    @property
    def action_dim(self):
        return 5

    def reset(self, random_start=True):
        self.battery.reset()
        self.diesel.reset()
        self.records = []
        self.steps = 0

        self.prev_batt_ac_kw = 0.0
        self.prev_diesel_kw = 0.0
        self.prev_dg_on = 0.0

        if random_start:
            max_start = max(len(self.df) - self.episode_len - self.forecast_horizon - 1, 1)
            self.idx = int(self.rng.integers(0, max_start))
        else:
            self.idx = 0
            
        self.state_buffer.clear()
        state = self._get_single_state(self.idx)
        for _ in range(self.seq_len):
            self.state_buffer.append(state)

        return self._get_state_sequence()

    def _pv_at_idx(self, idx):
        idx = min(max(idx, 0), len(self.df) - 1)
        row = self.df.iloc[idx]
        return self.pv.power_kw(
            float(row["ghi_wm2"]),
            float(row["cell_temp_c"]),
            t_seconds=idx * self.dt_s
        )

    def _forecast_features(self, idx):
        end_idx = min(idx + self.forecast_horizon + 1, len(self.df))
        future = self.df.iloc[idx + 1:end_idx]
        if len(future) == 0:
            return float(self.df.iloc[idx]["load_kw"]), self._pv_at_idx(idx)[0]
        load_fc = float(future["load_kw"].mean())
        pv_fc = float(np.mean([self._pv_at_idx(i)[0] for i in range(idx + 1, end_idx)]))
        return load_fc, pv_fc

    def _get_single_state(self, idx):
        row = self.df.iloc[idx]
        load_kw = float(row["load_kw"])
        pv_ac_kw = self._pv_at_idx(idx)[0]
        net_kw = load_kw - pv_ac_kw
        deficit_kw = max(net_kw, 0.0)
        surplus_kw = max(-net_kw, 0.0)

        load_fc, pv_fc = self._forecast_features(idx)
        ts = pd.to_datetime(row["timestamp"])
        hour = ts.hour + ts.minute / 60.0

        return np.array([
            load_kw / self.max_load,
            pv_ac_kw / self.max_pv,
            np.clip(net_kw / self.norm_power, -1.0, 1.0),
            deficit_kw / self.norm_power,
            surplus_kw / self.norm_power,
            self.battery.soc,
            self.battery.soh,
            self.prev_batt_ac_kw / self.norm_power,
            self.prev_diesel_kw / self.diesel.p_rated_kw,
            self.prev_dg_on,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            load_fc / self.max_load,
            pv_fc / self.max_pv,
        ], dtype=np.float32)

    def _get_state_sequence(self):
        if self.seq_len == 1:
            return self.state_buffer[-1]
        return np.concatenate(self.state_buffer, axis=0)

    def _decode_raw_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        diesel_frac = float((action[1] + 1.0) / 2.0)
        curt_frac = float((action[2] + 1.0) / 2.0)
        soc_target = float(self.battery.soc_min + ((action[3] + 1.0) / 2.0) * (self.battery.soc_max - self.battery.soc_min))
        
        return {
            "batt_norm": float(action[0]),
            "diesel_frac": diesel_frac,
            "curt_frac": curt_frac,
            "soc_target": soc_target,
            "dg_commit": 1 if action[4] > 0 else 0,
            "p_batt_ref_kw": float(action[0]) * max(self.battery.p_ch_max_kw, self.battery.p_dis_max_kw),
            "p_diesel_ref_kw": diesel_frac * self.diesel.p_rated_kw,
            "raw_action_0": float(action[0]),
            "raw_action_1": float(action[1]),
            "raw_action_2": float(action[2]),
            "raw_action_3": float(action[3]),
            "raw_action_4": float(action[4]),
        }

    def _fuzzy_action_shaping(self, decoded, fuzzy, load_kw, pv_ac_kw):
        shaped = dict(decoded)
        deficit_kw = max(load_kw - pv_ac_kw, 0.0)
        surplus_kw = max(pv_ac_kw - load_kw, 0.0)
        gain = self.fuzzy_action_gain

        p_batt = shaped["p_batt_ref_kw"]
        if surplus_kw > 1e-6:
            charge_push = fuzzy["charge_priority"] * min(surplus_kw, self.battery.p_ch_max_kw)
            p_batt = (1.0 - gain) * p_batt + gain * charge_push

        if deficit_kw > 1e-6:
            discharge_push = -fuzzy["discharge_priority"] * min(deficit_kw, self.battery.p_dis_max_kw)
            p_batt = (1.0 - gain) * p_batt + gain * discharge_push

        if p_batt < 0:
            p_batt *= fuzzy["discharge_priority"]
        if p_batt > 0:
            p_batt *= max(fuzzy["charge_priority"], 0.05)

        p_batt = float(np.clip(p_batt, -self.battery.p_dis_max_kw, self.battery.p_ch_max_kw))
        shaped["p_batt_ref_kw"] = p_batt
        shaped["batt_norm"] = float(np.clip(p_batt / max(self.battery.p_ch_max_kw, self.battery.p_dis_max_kw, 1e-9), -1.0, 1.0))

        p_diesel = shaped["p_diesel_ref_kw"]
        if deficit_kw <= 1e-6:
            p_diesel *= 0.05 * fuzzy["diesel_permission"]
            shaped["dg_commit"] = 0
        else:
            fuzzy_diesel_ref = fuzzy["diesel_permission"] * deficit_kw
            p_diesel = (1.0 - gain) * p_diesel + gain * fuzzy_diesel_ref
            if fuzzy["diesel_permission"] > 0.35 or p_diesel >= self.diesel.p_min_kw:
                shaped["dg_commit"] = 1
            else:
                shaped["dg_commit"] = 0

        p_diesel = float(np.clip(p_diesel, 0.0, self.diesel.p_rated_kw))
        shaped["p_diesel_ref_kw"] = p_diesel
        shaped["diesel_frac"] = float(np.clip(p_diesel / self.diesel.p_rated_kw, 0.0, 1.0))

        curt_frac = shaped["curt_frac"]
        if surplus_kw <= 1e-6:
            curt_frac = 0.0
        else:
            curt_frac = (1.0 - gain) * curt_frac + gain * fuzzy["curtail_permission"]
        shaped["curt_frac"] = float(np.clip(curt_frac, 0.0, 1.0))

        if fuzzy["forecast_balance_ratio"] < -0.05:
            desired_soc = 0.65
        elif fuzzy["forecast_balance_ratio"] > 0.05:
            desired_soc = 0.50
        else:
            desired_soc = 0.55

        shaped["soc_target"] = float(np.clip(
            (1.0 - 0.20 * gain) * shaped["soc_target"] + (0.20 * gain) * desired_soc,
            self.battery.soc_min,
            self.battery.soc_max
        ))

        return shaped

    def _safety_projection(self, decoded, fuzzy, load_kw, pv_ac_kw):
        soc = self.battery.soc
        max_charge_by_soc = self.battery.max_charge_kw_by_soc(self.dt_h)
        max_dis_by_soc = self.battery.max_discharge_kw_by_soc(self.dt_h)

        raw_curt_kw = float(np.clip(decoded["curt_frac"] * pv_ac_kw, 0.0, pv_ac_kw))
        pv_surplus_raw_kw = max(pv_ac_kw - load_kw, 0.0)
        unabsorbable_surplus_kw = max(pv_surplus_raw_kw - max_charge_by_soc, 0.0)

        if pv_surplus_raw_kw <= 1e-6:
            p_curt_ref_kw = 0.0
        elif soc >= self.battery.soc_max - 0.02:
            p_curt_ref_kw = min(raw_curt_kw, pv_surplus_raw_kw)
        else:
            fuzzy_curt_limit = fuzzy["curtail_permission"] * pv_surplus_raw_kw
            p_curt_ref_kw = min(raw_curt_kw, max(unabsorbable_surplus_kw, fuzzy_curt_limit))

        p_curt_ref_kw = float(np.clip(p_curt_ref_kw, 0.0, pv_ac_kw))
        pv_after_curt_kw = pv_ac_kw - p_curt_ref_kw
        surplus_after_pv_kw = max(pv_after_curt_kw - load_kw, 0.0)
        deficit_after_pv_kw = max(load_kw - pv_after_curt_kw, 0.0)

        p_batt_req_kw = decoded["p_batt_ref_kw"]
        if surplus_after_pv_kw > 1e-6:
            p_batt_req_kw = max(p_batt_req_kw, fuzzy["charge_priority"] * min(max_charge_by_soc, surplus_after_pv_kw))
        if deficit_after_pv_kw > 1e-6:
            p_batt_req_kw = min(p_batt_req_kw, -fuzzy["discharge_priority"] * min(max_dis_by_soc, deficit_after_pv_kw))

        if soc <= self.battery.soc_min + 0.015:
            p_batt_req_kw = max(p_batt_req_kw, 0.0)
        if soc >= self.battery.soc_max - 0.015:
            p_batt_req_kw = min(p_batt_req_kw, 0.0)

        if p_batt_req_kw >= 0:
            p_batt_safe_kw = max(min(p_batt_req_kw, max_charge_by_soc, surplus_after_pv_kw), 0.0)
        else:
            p_batt_safe_kw = min(-min(abs(p_batt_req_kw), max_dis_by_soc, deficit_after_pv_kw), 0.0)

        if decoded["dg_commit"] == 1:
            p_diesel_ref_kw = float(np.clip(decoded["p_diesel_ref_kw"] * max(fuzzy["diesel_permission"], 0.05), 0.0, self.diesel.p_rated_kw))
        else:
            p_diesel_ref_kw = 0.0

        return p_curt_ref_kw, pv_after_curt_kw, float(p_batt_safe_kw), p_diesel_ref_kw

    def step(self, action):
        row = self.df.iloc[self.idx]
        ts = row["timestamp"]

        load_kw = float(row["load_kw"])
        pv_ac_kw, pv_dc_kw, pitch, roll, motion_loss = self._pv_at_idx(self.idx)
        load_fc_kw, pv_fc_kw = self._forecast_features(self.idx)

        decoded_raw = self._decode_raw_action(action)
        fuzzy_info = self.fuzzy.infer(self.battery.soc, load_kw, pv_ac_kw, load_fc_kw, pv_fc_kw, self.prev_dg_on)
        decoded = self._fuzzy_action_shaping(decoded_raw, fuzzy_info, load_kw, pv_ac_kw)
        fuzzy_action_penalty = self.fuzzy.action_penalty(decoded, fuzzy_info, load_kw, pv_ac_kw)

        p_curt_ref_kw, pv_after_curt_kw, p_batt_cmd_ac_kw, p_diesel_safe_ref_kw = self._safety_projection(decoded, fuzzy_info, load_kw, pv_ac_kw)

        p_batt_internal_cmd, conv_loss_kw = self.converter.process_ac_to_internal(p_batt_cmd_ac_kw)
        p_batt_internal_actual, batt_deg_cost, soc, soh = self.battery.step(p_batt_internal_cmd, self.dt_h)
        p_batt_ac_kw = self.converter.internal_to_ac(p_batt_internal_actual)

        batt_charge_kw = max(p_batt_ac_kw, 0.0)
        batt_discharge_kw = max(-p_batt_ac_kw, 0.0)

        supply_without_diesel_kw = pv_after_curt_kw - p_batt_ac_kw - conv_loss_kw
        residual_deficit_kw = max(load_kw - supply_without_diesel_kw, 0.0)

        optional_diesel_reserve_kw = 0.10 * p_diesel_safe_ref_kw if residual_deficit_kw > 1e-6 else 0.0
        p_diesel_cmd_kw = residual_deficit_kw + optional_diesel_reserve_kw
        p_diesel_cmd_kw *= max(fuzzy_info["diesel_permission"], 0.05) if residual_deficit_kw > 1e-6 else 0.0

        if residual_deficit_kw > 1e-6 and soc <= self.battery.soc_min + 0.03:
            p_diesel_cmd_kw = max(p_diesel_cmd_kw, residual_deficit_kw)

        p_diesel_cmd_kw = float(np.clip(p_diesel_cmd_kw, 0.0, self.diesel.p_rated_kw))
        p_diesel_kw, fuel_l, dg_wear_cost, dg_status, dg_start = self.diesel.dispatch(p_diesel_cmd_kw, self.dt_h)

        supply_kw = pv_after_curt_kw + p_diesel_kw - p_batt_ac_kw - conv_loss_kw
        imbalance_kw = load_kw - supply_kw

        unmet_kw = max(imbalance_kw, 0.0)
        extra_curtail_kw = max(-imbalance_kw, 0.0)
        curtail_kw = p_curt_ref_kw + extra_curtail_kw

        served_kw = load_kw - unmet_kw
        line_loss_kw, i_line_A, v_bus_V, v_drop_V, current_violation = self.line.calc(served_kw)

        soc_violation = max(self.battery.soc_min - soc, 0.0) + max(soc - self.battery.soc_max, 0.0)
        diesel_violation = max(p_diesel_kw - self.diesel.p_rated_kw, 0.0)
        batt_violation = max(abs(p_batt_ac_kw) - self.converter.p_rated_kw, 0.0)

        c_constraint = 1000.0 * soc_violation + 100.0 * float(current_violation) + 100.0 * diesel_violation + 100.0 * batt_violation
        c_fuzzy = self.fuzzy_penalty_weight * fuzzy_action_penalty

        reward_raw = -(
            fuel_l + self.diesel.start_cost * dg_start + dg_wear_cost + batt_deg_cost + 
            2.0 * curtail_kw * self.dt_h + 800.0 * unmet_kw * self.dt_h + 
            25.0 * p_diesel_kw * self.dt_h + 0.25 * abs(soc - decoded["soc_target"]) + 
            c_constraint + c_fuzzy
        ) + (
            5.0 * min(batt_discharge_kw, max(load_kw - pv_after_curt_kw, 0.0)) * self.dt_h +
            2.0 * min(batt_charge_kw, max(pv_after_curt_kw - load_kw, 0.0)) * self.dt_h +
            2.0 * max(min(pv_after_curt_kw, load_kw + batt_charge_kw), 0.0) * self.dt_h
        )

        reward = float(np.clip(reward_raw / max(self.reward_scale, 1e-9), self.reward_clip[0], self.reward_clip[1]))

        self.records.append({
            "timestamp": ts, "load_kw": load_kw, "pv_ac_kw": pv_ac_kw, "pv_after_curt_kw": pv_after_curt_kw,
            "p_batt_ref_raw_kw": decoded_raw["p_batt_ref_kw"], "p_batt_ref_fuzzy_kw": decoded["p_batt_ref_kw"],
            "p_batt_cmd_safe_kw": p_batt_cmd_ac_kw, "batt_ac_kw": p_batt_ac_kw,
            "p_diesel_ref_raw_kw": decoded_raw["p_diesel_ref_kw"], "p_diesel_ref_fuzzy_kw": decoded["p_diesel_ref_kw"],
            "p_diesel_cmd_safe_kw": p_diesel_cmd_kw, "diesel_kw": p_diesel_kw,
            "raw_soc_target": decoded_raw["soc_target"], "fuzzy_soc_target": decoded["soc_target"], "soc": soc,
            "unmet_kw": unmet_kw, "curtail_kw": curtail_kw, "fuzzy_action_penalty": fuzzy_action_penalty,
            "reward": reward, "fuel_l": fuel_l, "batt_discharge_kw_positive": batt_discharge_kw,
            "batt_charge_kw_positive": batt_charge_kw, "fuzzy_charge_priority": fuzzy_info["charge_priority"],
            "fuzzy_discharge_priority": fuzzy_info["discharge_priority"], "fuzzy_diesel_permission": fuzzy_info["diesel_permission"],
            "fuzzy_curtail_permission": fuzzy_info["curtail_permission"], "fuzzy_soc_protection": fuzzy_info["soc_protection"],
            "dg_wear_cost": dg_wear_cost, "batt_deg_cost": batt_deg_cost, "conv_loss_kw": conv_loss_kw, "line_loss_kw": line_loss_kw,
            "soh": soh, "diesel_start": dg_start, "current_violation": current_violation, "soc_violation": soc_violation,
            "diesel_violation": diesel_violation, "batt_violation": batt_violation, "v_bus_V": v_bus_V, "cost_fuzzy": c_fuzzy,
            "reward_raw": reward_raw,
        })

        self.prev_batt_ac_kw = p_batt_ac_kw
        self.prev_diesel_kw = p_diesel_kw
        self.prev_dg_on = 1.0 if p_diesel_kw > 1e-6 else 0.0

        self.idx += 1
        self.steps += 1
        
        self.state_buffer.append(self._get_single_state(self.idx) if self.idx < len(self.df) else np.zeros(self.state_dim, dtype=np.float32))

        done = (self.steps >= self.episode_len or self.idx >= len(self.df) - self.forecast_horizon - 1)
        next_state = self._get_state_sequence() if not done else np.zeros(self.state_dim * self.seq_len, dtype=np.float32)

        return next_state, reward, done, {}

    def get_results(self):
        return pd.DataFrame(self.records).set_index("timestamp")
