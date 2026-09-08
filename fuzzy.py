# ================================================================
# Fuzzy-Embedded Safety-Constrained Hierarchical SAC EMS
# Floating PV-Battery-Diesel Microgrid
#
# Training: load_with_weather.csv from 2016-01-01 to 2018-12-31
# Testing : target_load.csv
#
# Main features:
#   1) Soft Actor-Critic (SAC)
#   2) Fuzzy logic embedded into actor/action command shaping
#   3) Fuzzy-informed safety projection layer
#   4) Fuzzy rule-violation penalty in reward function
#   5) Battery/PV-priority, diesel-last operation
#
# Required input columns:
#   timestamp / DateTime / Time / Date
#   load_kw / Load_kW / Load / RealPower / Power_kW
#   ghi_wm2 / GHI / Irradiance / ALLSKY_SFC_SW_DWN
#   cell_temp_c / Temperature / Temp_Air_C / T2M optional
# ================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ================================================================
# 0. Utilities
# ================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_column(df, candidates, name):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).strip().lower() in normalized:
            return normalized[str(c).strip().lower()]
    raise ValueError(f"No {name} column found. Available columns: {df.columns.tolist()}")


def load_training_data_from_load_with_weather(
    training_file="load_with_weather.csv",
    start_date="2016-01-01",
    end_date="2018-12-31 23:59:59"
):
    df = pd.read_csv(training_file)

    ts_col = detect_column(
        df,
        ["timestamp", "Timestamp", "DateTime", "Datetime", "date_time", "Date Time", "Time", "Date"],
        "timestamp"
    )
    load_col = detect_column(
        df,
        ["load_kw", "Load_kW", "Load", "load", "Power_kW", "RealPower", "kW", "KW"],
        "load"
    )
    ghi_col = detect_column(
        df,
        ["ghi_wm2", "GHI_Wm2", "GHI", "Irradiance", "irradiance", "ALLSKY_SFC_SW_DWN"],
        "irradiance"
    )

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["load_kw"] = pd.to_numeric(df[load_col], errors="coerce")
    df["ghi_wm2"] = pd.to_numeric(df[ghi_col], errors="coerce")

    temp_col = None
    for c in ["cell_temp_c", "Temp_Air_C", "T2M", "temperature", "Temperature", "temp_c"]:
        if c in df.columns:
            temp_col = c
            break

    if temp_col is None:
        df["cell_temp_c"] = 25.0
    else:
        df["cell_temp_c"] = pd.to_numeric(df[temp_col], errors="coerce").fillna(25.0)

    df = df.dropna(subset=["timestamp", "load_kw", "ghi_wm2", "cell_temp_c"])
    df["load_kw"] = df["load_kw"].clip(lower=0)
    df["ghi_wm2"] = df["ghi_wm2"].clip(lower=0)

    df = df[
        (df["timestamp"] >= pd.to_datetime(start_date)) &
        (df["timestamp"] <= pd.to_datetime(end_date))
    ]

    df = df[["timestamp", "load_kw", "ghi_wm2", "cell_temp_c"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Training dataframe is empty. Check load_with_weather.csv date range.")

    print("\n===== TRAINING DATA LOADED =====")
    print(f"File   : {training_file}")
    print(f"Samples: {len(df)}")
    print(f"Range  : {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def load_test_operation_data(test_file="target_load.csv"):
    df = pd.read_csv(test_file)

    ts_col = detect_column(
        df,
        ["timestamp", "Timestamp", "DateTime", "Datetime", "date_time", "Date Time", "Time", "Date"],
        "timestamp"
    )
    load_col = detect_column(
        df,
        ["load_kw", "Load_kW", "Load", "load", "RealPower", "Power_kW", "power_kw", "kW", "KW"],
        "load"
    )
    ghi_col = detect_column(
        df,
        ["ghi_wm2", "GHI_Wm2", "GHI", "Irradiance", "irradiance", "ALLSKY_SFC_SW_DWN"],
        "irradiance"
    )

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["load_kw"] = pd.to_numeric(df[load_col], errors="coerce")
    df["ghi_wm2"] = pd.to_numeric(df[ghi_col], errors="coerce")

    temp_col = None
    for c in ["cell_temp_c", "Temp_Air_C", "T2M", "temperature", "Temperature", "temp_c"]:
        if c in df.columns:
            temp_col = c
            break

    if temp_col is None:
        df["cell_temp_c"] = 25.0
    else:
        df["cell_temp_c"] = pd.to_numeric(df[temp_col], errors="coerce").fillna(25.0)

    df = df.dropna(subset=["timestamp", "load_kw", "ghi_wm2", "cell_temp_c"])
    df["load_kw"] = df["load_kw"].clip(lower=0)
    df["ghi_wm2"] = df["ghi_wm2"].clip(lower=0)

    df = df[["timestamp", "load_kw", "ghi_wm2", "cell_temp_c"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Target test dataframe is empty. Check target_load.csv.")

    print("\n===== TARGET TEST DATA LOADED =====")
    print(f"File   : {test_file}")
    print(f"Samples: {len(df)}")
    print(f"Range  : {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


# ================================================================
# 1. Physical Component Models
# ================================================================
@dataclass
class FloatingPVArray:
    p_stc_kw: float = 250.0
    gamma_pmp: float = -0.004
    t_ref: float = 25.0
    g_ref: float = 1000.0
    pitch_amp_deg: float = 3.0
    roll_amp_deg: float = 3.0
    wave_period_s: float = 8.0
    mismatch_coeff: float = 0.015
    inverter_eff: float = 0.97

    def power_kw(self, ghi, cell_temp, t_seconds=0.0):
        pitch = self.pitch_amp_deg * np.sin(2 * np.pi * t_seconds / self.wave_period_s)
        roll = self.roll_amp_deg * np.cos(2 * np.pi * t_seconds / self.wave_period_s)

        theta_rms = np.sqrt(pitch ** 2 + roll ** 2)
        motion_loss = np.clip(1 - self.mismatch_coeff * theta_rms, 0.75, 1.0)

        ghi_eff = ghi * motion_loss

        p_dc = self.p_stc_kw * (ghi_eff / self.g_ref) * (
            1 + self.gamma_pmp * (cell_temp - self.t_ref)
        )
        p_dc = np.clip(p_dc, 0.0, self.p_stc_kw)
        p_ac = self.inverter_eff * p_dc

        return float(p_ac), float(p_dc), float(pitch), float(roll), float(motion_loss)


@dataclass
class DegradationBattery:
    e_capacity_kwh: float = 1000.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.60
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    c_rate: float = 0.5
    replacement_cost_per_kwh: float = 180.0
    cycle_life: float = 4000.0
    calendar_fade_per_day: float = 0.00002

    soc: float = field(init=False)
    soh: float = field(default=1.0, init=False)
    throughput_kwh: float = field(default=0.0, init=False)
    degradation_cost: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.soc = float(np.clip(self.soc_init, self.soc_min, self.soc_max))
        self.soh = 1.0
        self.throughput_kwh = 0.0
        self.degradation_cost = 0.0

    @property
    def usable_capacity_kwh(self):
        return self.e_capacity_kwh * self.soh

    @property
    def p_ch_max_kw(self):
        return self.c_rate * self.usable_capacity_kwh

    @property
    def p_dis_max_kw(self):
        return self.c_rate * self.usable_capacity_kwh

    def max_charge_kw_by_soc(self, dt_h):
        e_room = max((self.soc_max - self.soc) * self.usable_capacity_kwh, 0.0)
        return min(self.p_ch_max_kw, e_room / max(self.eta_ch * dt_h, 1e-9))

    def max_discharge_kw_by_soc(self, dt_h):
        e_avail = max((self.soc - self.soc_min) * self.usable_capacity_kwh, 0.0)
        return min(self.p_dis_max_kw, e_avail * self.eta_dis / max(dt_h, 1e-9))

    def step(self, p_batt_internal_kw, dt_h):
        if p_batt_internal_kw >= 0:
            p_req = min(p_batt_internal_kw, self.p_ch_max_kw)
            e_to_batt = min(
                p_req * self.eta_ch * dt_h,
                max((self.soc_max - self.soc) * self.usable_capacity_kwh, 0.0)
            )
            self.soc += e_to_batt / max(self.usable_capacity_kwh, 1e-9)
            p_actual = e_to_batt / max(self.eta_ch * dt_h, 1e-9)
            throughput = p_actual * dt_h
        else:
            p_req = min(-p_batt_internal_kw, self.p_dis_max_kw)
            e_from_batt = min(
                p_req / self.eta_dis * dt_h,
                max((self.soc - self.soc_min) * self.usable_capacity_kwh, 0.0)
            )
            self.soc -= e_from_batt / max(self.usable_capacity_kwh, 1e-9)
            p_actual = -e_from_batt * self.eta_dis / max(dt_h, 1e-9)
            throughput = -p_actual * dt_h

        self.throughput_kwh += throughput

        cycle_fade = throughput / max(2 * self.e_capacity_kwh * self.cycle_life, 1e-9)
        calendar_fade = self.calendar_fade_per_day * (dt_h / 24.0)
        self.soh = max(0.70, self.soh - cycle_fade - calendar_fade)

        deg_cost = throughput * self.replacement_cost_per_kwh / max(2 * self.cycle_life, 1e-9)
        self.degradation_cost += deg_cost

        return float(p_actual), float(deg_cost), float(self.soc), float(self.soh)


@dataclass
class BatteryConverter:
    eta_nom: float = 0.96
    p_rated_kw: float = 500.0
    standby_loss_kw: float = 0.0

    def efficiency(self, p_kw):
        if abs(p_kw) < 1e-9:
            return self.eta_nom

        loading = min(abs(p_kw) / self.p_rated_kw, 1.0)
        return float(np.clip(self.eta_nom - 0.04 * (1 - loading), 0.90, self.eta_nom))

    def process_ac_to_internal(self, p_batt_ac_cmd_kw):
        if abs(p_batt_ac_cmd_kw) < 1e-9:
            return 0.0, self.standby_loss_kw

        eta = self.efficiency(p_batt_ac_cmd_kw)

        if p_batt_ac_cmd_kw >= 0:
            p_internal = p_batt_ac_cmd_kw * eta
            loss_kw = p_batt_ac_cmd_kw - p_internal
        else:
            p_internal = p_batt_ac_cmd_kw / eta
            loss_kw = abs(p_internal) - abs(p_batt_ac_cmd_kw)

        return float(p_internal), float(loss_kw)

    def internal_to_ac(self, p_batt_internal_actual_kw):
        if abs(p_batt_internal_actual_kw) < 1e-9:
            return 0.0

        eta = self.efficiency(p_batt_internal_actual_kw)

        if p_batt_internal_actual_kw >= 0:
            return float(p_batt_internal_actual_kw / eta)
        else:
            return float(p_batt_internal_actual_kw * eta)


@dataclass
class DynamicDieselGen:
    p_rated_kw: float = 120.0
    p_min_kw: float = 30.0
    sfc_rated: float = 0.28
    ramp_rate_kw_per_min: float = 300.0
    start_fuel_l: float = 0.8
    start_cost: float = 2.0
    maintenance_cost_per_h: float = 3.0

    running: bool = False
    p_prev_kw: float = 0.0
    starts: int = 0

    def reset(self):
        self.running = False
        self.p_prev_kw = 0.0
        self.starts = 0

    def dispatch(self, p_req_kw, dt_h):
        if p_req_kw <= 1e-6:
            self.running = False
            self.p_prev_kw = 0.0
            return 0.0, 0.0, 0.0, "OFF", 0

        start_cost = 0.0
        start_fuel = 0.0
        started = 0

        if not self.running:
            self.running = True
            self.starts += 1
            start_cost = self.start_cost
            start_fuel = self.start_fuel_l
            started = 1

        p_target = np.clip(p_req_kw, self.p_min_kw, self.p_rated_kw)
        ramp_limit = self.ramp_rate_kw_per_min * dt_h * 60.0

        if self.p_prev_kw <= 1e-6:
            p_actual = p_target
        else:
            p_actual = np.clip(p_target, self.p_prev_kw - ramp_limit, self.p_prev_kw + ramp_limit)

        p_actual = np.clip(p_actual, self.p_min_kw, self.p_rated_kw)

        loading = p_actual / self.p_rated_kw
        sfc = self.sfc_rated * (1 + 0.25 * (1 - loading))

        fuel_l = sfc * p_actual * dt_h + start_fuel
        wear_cost = start_cost + self.maintenance_cost_per_h * dt_h

        self.p_prev_kw = p_actual

        return float(p_actual), float(fuel_l), float(wear_cost), "ON", int(started)


@dataclass
class LVLineAdvanced:
    r_ohm: float = 0.02
    x_ohm: float = 0.01
    v_ll: float = 400.0
    i_max_A: float = 250.0
    pf: float = 0.95

    def calc(self, p_flow_kw):
        if p_flow_kw <= 0:
            return 0.0, 0.0, self.v_ll, 0.0, False

        i = (p_flow_kw * 1e3) / (np.sqrt(3) * self.v_ll * self.pf)
        loss_kw = 3 * i ** 2 * self.r_ohm / 1e3

        sin_phi = np.sqrt(max(1 - self.pf ** 2, 0.0))
        v_drop = np.sqrt(3) * i * (self.r_ohm * self.pf + self.x_ohm * sin_phi)
        v_bus = self.v_ll - v_drop

        return float(loss_kw), float(i), float(v_bus), float(v_drop), bool(i > self.i_max_A)


# ================================================================
# 2. Fuzzy Logic Embedded Layer
# ================================================================
class FuzzyEmbeddedSupervisor:
    """
    This is not a separate baseline FLC.

    It is embedded inside the SAC controller in three places:
      1) Actor/action command shaping
      2) Safety projection coefficient generation
      3) Reward penalty calculation

    Outputs are continuous priority factors in [0, 1].
    """

    def __init__(
        self,
        soc_min=0.10,
        soc_max=0.90,
        soc_critical=0.15,
        soc_low=0.30,
        soc_mid=0.55,
        soc_high=0.78,
        soc_full=0.88
    ):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_critical = soc_critical
        self.soc_low = soc_low
        self.soc_mid = soc_mid
        self.soc_high = soc_high
        self.soc_full = soc_full

    @staticmethod
    def trimf(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return float((x - a) / max(b - a, 1e-9))
        return float((c - x) / max(c - b, 1e-9))

    @staticmethod
    def trapmf(x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return float((x - a) / max(b - a, 1e-9))
        return float((d - x) / max(d - c, 1e-9))

    def fuzzify_soc(self, soc):
        return {
            "critical": self.trapmf(soc, 0.00, 0.00, self.soc_min, self.soc_critical),
            "low": self.trimf(soc, self.soc_min, self.soc_low, self.soc_mid),
            "medium": self.trimf(soc, self.soc_low, self.soc_mid, self.soc_high),
            "high": self.trimf(soc, self.soc_mid, self.soc_high, self.soc_max),
            "full": self.trapmf(soc, self.soc_high, self.soc_full, self.soc_max, self.soc_max),
        }

    def fuzzify_balance(self, balance_ratio):
        """
        balance_ratio = (PV - Load) / max(Load, rated_norm)
        negative = deficit
        positive = surplus
        """
        return {
            "large_deficit": self.trapmf(balance_ratio, -2.0, -1.0, -0.55, -0.25),
            "small_deficit": self.trimf(balance_ratio, -0.60, -0.25, 0.0),
            "balanced": self.trimf(balance_ratio, -0.12, 0.0, 0.12),
            "small_surplus": self.trimf(balance_ratio, 0.0, 0.25, 0.60),
            "large_surplus": self.trapmf(balance_ratio, 0.25, 0.55, 1.0, 2.0),
        }

    def fuzzify_forecast(self, forecast_balance_ratio):
        return {
            "future_deficit": self.trapmf(forecast_balance_ratio, -2.0, -1.0, -0.35, -0.08),
            "future_balanced": self.trimf(forecast_balance_ratio, -0.15, 0.0, 0.15),
            "future_surplus": self.trapmf(forecast_balance_ratio, 0.08, 0.35, 1.0, 2.0),
        }

    @staticmethod
    def weighted_average(rules, default=0.0):
        total_w = sum(w for w, _ in rules)
        if total_w <= 1e-9:
            return float(default)
        return float(sum(w * y for w, y in rules) / total_w)

    def infer(
        self,
        soc,
        load_kw,
        pv_kw,
        load_fc_kw,
        pv_fc_kw,
        prev_dg_on=0.0
    ):
        load_base = max(load_kw, 1e-6)
        balance_ratio = np.clip((pv_kw - load_kw) / load_base, -2.0, 2.0)
        forecast_ratio = np.clip((pv_fc_kw - load_fc_kw) / max(load_fc_kw, 1e-6), -2.0, 2.0)

        mu_soc = self.fuzzify_soc(soc)
        mu_bal = self.fuzzify_balance(balance_ratio)
        mu_fc = self.fuzzify_forecast(forecast_ratio)

        # Battery charging priority:
        # high when PV surplus exists and SOC is not high/full
        charge_rules = [
            (min(mu_bal["large_surplus"], mu_soc["critical"]), 1.00),
            (min(mu_bal["large_surplus"], mu_soc["low"]), 0.95),
            (min(mu_bal["large_surplus"], mu_soc["medium"]), 0.80),
            (min(mu_bal["large_surplus"], mu_soc["high"]), 0.35),
            (min(mu_bal["large_surplus"], mu_soc["full"]), 0.00),

            (min(mu_bal["small_surplus"], mu_soc["critical"]), 0.90),
            (min(mu_bal["small_surplus"], mu_soc["low"]), 0.80),
            (min(mu_bal["small_surplus"], mu_soc["medium"]), 0.55),
            (min(mu_bal["small_surplus"], mu_soc["high"]), 0.20),
            (min(mu_bal["small_surplus"], mu_soc["full"]), 0.00),

            (min(mu_fc["future_deficit"], mu_soc["low"]), 0.30),
            (min(mu_fc["future_surplus"], mu_soc["low"]), 0.85),
        ]

        # Battery discharge priority:
        # high during deficit when SOC is medium/high, low when SOC is low/critical
        discharge_rules = [
            (min(mu_bal["large_deficit"], mu_soc["critical"]), 0.00),
            (min(mu_bal["large_deficit"], mu_soc["low"]), 0.25),
            (min(mu_bal["large_deficit"], mu_soc["medium"]), 0.75),
            (min(mu_bal["large_deficit"], mu_soc["high"]), 1.00),
            (min(mu_bal["large_deficit"], mu_soc["full"]), 1.00),

            (min(mu_bal["small_deficit"], mu_soc["critical"]), 0.00),
            (min(mu_bal["small_deficit"], mu_soc["low"]), 0.35),
            (min(mu_bal["small_deficit"], mu_soc["medium"]), 0.80),
            (min(mu_bal["small_deficit"], mu_soc["high"]), 1.00),
            (min(mu_bal["small_deficit"], mu_soc["full"]), 1.00),

            (min(mu_fc["future_deficit"], mu_soc["high"]), 0.65),
            (min(mu_fc["future_deficit"], mu_soc["medium"]), 0.50),
        ]

        # Diesel permission:
        # high when deficit and SOC is low/critical, lower when battery can support
        diesel_rules = [
            (min(mu_bal["large_deficit"], mu_soc["critical"]), 1.00),
            (min(mu_bal["large_deficit"], mu_soc["low"]), 0.85),
            (min(mu_bal["large_deficit"], mu_soc["medium"]), 0.45),
            (min(mu_bal["large_deficit"], mu_soc["high"]), 0.15),

            (min(mu_bal["small_deficit"], mu_soc["critical"]), 0.85),
            (min(mu_bal["small_deficit"], mu_soc["low"]), 0.60),
            (min(mu_bal["small_deficit"], mu_soc["medium"]), 0.20),
            (min(mu_bal["small_deficit"], mu_soc["high"]), 0.05),

            (min(mu_fc["future_deficit"], mu_soc["low"]), 0.80),
            (min(mu_fc["future_deficit"], mu_soc["critical"]), 1.00),
        ]

        # Keep diesel on softly if it was already on and deficit still exists.
        if prev_dg_on > 0.5:
            diesel_rules.append((mu_bal["small_deficit"], 0.35))
            diesel_rules.append((mu_bal["large_deficit"], 0.60))

        # Curtailment permission:
        # high only when surplus exists and SOC is high/full
        curtail_rules = [
            (min(mu_bal["large_surplus"], mu_soc["full"]), 1.00),
            (min(mu_bal["large_surplus"], mu_soc["high"]), 0.65),
            (min(mu_bal["large_surplus"], mu_soc["medium"]), 0.20),
            (min(mu_bal["large_surplus"], mu_soc["low"]), 0.05),

            (min(mu_bal["small_surplus"], mu_soc["full"]), 0.80),
            (min(mu_bal["small_surplus"], mu_soc["high"]), 0.35),
            (min(mu_bal["small_surplus"], mu_soc["medium"]), 0.10),
            (min(mu_bal["small_surplus"], mu_soc["low"]), 0.00),
        ]

        # SOC protection:
        # high when SOC is near lower/upper boundaries
        soc_protect_rules = [
            (mu_soc["critical"], 1.00),
            (mu_soc["low"], 0.65),
            (mu_soc["medium"], 0.15),
            (mu_soc["high"], 0.35),
            (mu_soc["full"], 1.00),
        ]

        charge_priority = self.weighted_average(charge_rules, default=0.0)
        discharge_priority = self.weighted_average(discharge_rules, default=0.0)
        diesel_permission = self.weighted_average(diesel_rules, default=0.0)
        curtail_permission = self.weighted_average(curtail_rules, default=0.0)
        soc_protection = self.weighted_average(soc_protect_rules, default=0.2)

        return {
            "balance_ratio": float(balance_ratio),
            "forecast_balance_ratio": float(forecast_ratio),
            "charge_priority": float(np.clip(charge_priority, 0.0, 1.0)),
            "discharge_priority": float(np.clip(discharge_priority, 0.0, 1.0)),
            "diesel_permission": float(np.clip(diesel_permission, 0.0, 1.0)),
            "curtail_permission": float(np.clip(curtail_permission, 0.0, 1.0)),
            "soc_protection": float(np.clip(soc_protection, 0.0, 1.0)),
            "mu_soc_critical": float(mu_soc["critical"]),
            "mu_soc_low": float(mu_soc["low"]),
            "mu_soc_medium": float(mu_soc["medium"]),
            "mu_soc_high": float(mu_soc["high"]),
            "mu_soc_full": float(mu_soc["full"]),
            "mu_large_deficit": float(mu_bal["large_deficit"]),
            "mu_small_deficit": float(mu_bal["small_deficit"]),
            "mu_balanced": float(mu_bal["balanced"]),
            "mu_small_surplus": float(mu_bal["small_surplus"]),
            "mu_large_surplus": float(mu_bal["large_surplus"]),
        }

    def action_penalty(self, decoded, fuzzy, load_kw, pv_kw):
        """
        Penalizes actor commands that oppose fuzzy expert knowledge.
        This affects critic learning through reward shaping.
        """
        deficit_kw = max(load_kw - pv_kw, 0.0)
        surplus_kw = max(pv_kw - load_kw, 0.0)

        p_batt_ref = decoded["p_batt_ref_kw"]
        p_diesel_ref = decoded["p_diesel_ref_kw"]
        curt_frac = decoded["curt_frac"]

        penalty = 0.0

        # Battery discharge while fuzzy says protect SOC.
        if p_batt_ref < 0:
            penalty += fuzzy["soc_protection"] * (1.0 - fuzzy["discharge_priority"]) * abs(p_batt_ref) / 500.0

        # Battery charge while fuzzy says curtailment/full SOC is preferred.
        if p_batt_ref > 0 and fuzzy["charge_priority"] < 0.15:
            penalty += (0.15 - fuzzy["charge_priority"]) * p_batt_ref / 500.0

        # Diesel requested when not needed or fuzzy does not permit it.
        if p_diesel_ref > 1e-6:
            if deficit_kw <= 1e-6:
                penalty += 0.75 * p_diesel_ref / 120.0
            penalty += (1.0 - fuzzy["diesel_permission"]) * p_diesel_ref / 120.0

        # Curtailment requested when usable surplus and battery should charge.
        if surplus_kw > 1e-6 and curt_frac > 0:
            penalty += (1.0 - fuzzy["curtail_permission"]) * curt_frac

        return float(max(penalty, 0.0))


# ================================================================
# 3. Fuzzy-Embedded SAC Microgrid Environment
# ================================================================
class FuzzyEmbeddedSACMicrogridEnv:
    def __init__(
        self,
        df,
        dt_h=0.25,
        episode_len=96,
        forecast_horizon=4,
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

        return self._get_state()

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

    def _get_state(self):
        row = self.df.iloc[self.idx]

        load_kw = float(row["load_kw"])
        pv_ac_kw = self._pv_at_idx(self.idx)[0]
        net_kw = load_kw - pv_ac_kw
        deficit_kw = max(net_kw, 0.0)
        surplus_kw = max(-net_kw, 0.0)

        load_fc, pv_fc = self._forecast_features(self.idx)

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

    def _decode_raw_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)

        diesel_frac = float((action[1] + 1.0) / 2.0)
        curt_frac = float((action[2] + 1.0) / 2.0)

        soc_target = float(
            self.battery.soc_min +
            ((action[3] + 1.0) / 2.0) * (self.battery.soc_max - self.battery.soc_min)
        )

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
        """
        Embedded fuzzy action command shaping.

        This modifies SAC actor commands before safety projection.
        SAC is still the main optimizer, but fuzzy logic bends actions toward
        interpretable and physically desirable operation.
        """
        shaped = dict(decoded)

        deficit_kw = max(load_kw - pv_ac_kw, 0.0)
        surplus_kw = max(pv_ac_kw - load_kw, 0.0)

        gain = self.fuzzy_action_gain

        # Battery command shaping
        p_batt = shaped["p_batt_ref_kw"]

        # During surplus, encourage charging according to fuzzy charge priority.
        if surplus_kw > 1e-6:
            charge_push = fuzzy["charge_priority"] * min(surplus_kw, self.battery.p_ch_max_kw)
            p_batt = (1.0 - gain) * p_batt + gain * charge_push

        # During deficit, encourage discharge according to fuzzy discharge priority.
        if deficit_kw > 1e-6:
            discharge_push = -fuzzy["discharge_priority"] * min(deficit_kw, self.battery.p_dis_max_kw)
            p_batt = (1.0 - gain) * p_batt + gain * discharge_push

        # Protect SOC by damping discharge at low SOC and damping charge at full SOC.
        if p_batt < 0:
            p_batt *= fuzzy["discharge_priority"]
        if p_batt > 0:
            p_batt *= max(fuzzy["charge_priority"], 0.05)

        p_batt = float(np.clip(p_batt, -self.battery.p_dis_max_kw, self.battery.p_ch_max_kw))
        shaped["p_batt_ref_kw"] = p_batt
        shaped["batt_norm"] = float(np.clip(p_batt / max(self.battery.p_ch_max_kw, self.battery.p_dis_max_kw, 1e-9), -1.0, 1.0))

        # Diesel command shaping
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

        # Curtailment command shaping
        curt_frac = shaped["curt_frac"]

        if surplus_kw <= 1e-6:
            curt_frac = 0.0
        else:
            curt_frac = (1.0 - gain) * curt_frac + gain * fuzzy["curtail_permission"]

        curt_frac = float(np.clip(curt_frac, 0.0, 1.0))
        shaped["curt_frac"] = curt_frac

        # SOC target shaping
        if fuzzy["forecast_balance_ratio"] < -0.05:
            # future deficit: prefer higher SOC
            desired_soc = 0.65
        elif fuzzy["forecast_balance_ratio"] > 0.05:
            # future surplus: leave some room for charging
            desired_soc = 0.50
        else:
            desired_soc = 0.55

        shaped["soc_target"] = float(
            np.clip(
                (1.0 - 0.20 * gain) * shaped["soc_target"] + (0.20 * gain) * desired_soc,
                self.battery.soc_min,
                self.battery.soc_max
            )
        )

        return shaped

    def _safety_projection(self, decoded, fuzzy, load_kw, pv_ac_kw):
        """
        Fuzzy-informed safety projection.

        Compared with fixed-rule projection, this uses fuzzy priorities:
          charge_priority
          discharge_priority
          diesel_permission
          curtail_permission
          soc_protection
        """

        soc = self.battery.soc

        max_charge_by_soc = self.battery.max_charge_kw_by_soc(self.dt_h)
        max_dis_by_soc = self.battery.max_discharge_kw_by_soc(self.dt_h)

        # --------------------------------------------------------
        # PV curtailment-last with fuzzy permission
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Fuzzy battery-priority dispatch
        # --------------------------------------------------------
        p_batt_req_kw = decoded["p_batt_ref_kw"]

        if surplus_after_pv_kw > 1e-6:
            desired_charge_kw = fuzzy["charge_priority"] * min(max_charge_by_soc, surplus_after_pv_kw)
            p_batt_req_kw = max(p_batt_req_kw, desired_charge_kw)

        if deficit_after_pv_kw > 1e-6:
            desired_discharge_kw = fuzzy["discharge_priority"] * min(max_dis_by_soc, deficit_after_pv_kw)
            p_batt_req_kw = min(p_batt_req_kw, -desired_discharge_kw)

        # SOC hard protection
        if soc <= self.battery.soc_min + 0.015:
            p_batt_req_kw = max(p_batt_req_kw, 0.0)

        if soc >= self.battery.soc_max - 0.015:
            p_batt_req_kw = min(p_batt_req_kw, 0.0)

        if p_batt_req_kw >= 0:
            p_batt_safe_kw = min(p_batt_req_kw, max_charge_by_soc, surplus_after_pv_kw)
            p_batt_safe_kw = max(p_batt_safe_kw, 0.0)
        else:
            p_batt_safe_kw = -min(abs(p_batt_req_kw), max_dis_by_soc, deficit_after_pv_kw)
            p_batt_safe_kw = min(p_batt_safe_kw, 0.0)

        # --------------------------------------------------------
        # Fuzzy diesel permission
        # --------------------------------------------------------
        if decoded["dg_commit"] == 1:
            p_diesel_ref_kw = float(
                np.clip(
                    decoded["p_diesel_ref_kw"] * max(fuzzy["diesel_permission"], 0.05),
                    0.0,
                    self.diesel.p_rated_kw
                )
            )
        else:
            p_diesel_ref_kw = 0.0

        return p_curt_ref_kw, pv_after_curt_kw, float(p_batt_safe_kw), p_diesel_ref_kw

    def step(self, action):
        row = self.df.iloc[self.idx]
        ts = row["timestamp"]

        load_kw = float(row["load_kw"])
        pv_ac_kw, pv_dc_kw, pitch, roll, motion_loss = self._pv_at_idx(self.idx)
        load_fc_kw, pv_fc_kw = self._forecast_features(self.idx)

        # SAC actor output
        decoded_raw = self._decode_raw_action(action)

        # Fuzzy inference from current state and forecasts
        fuzzy_info = self.fuzzy.infer(
            soc=self.battery.soc,
            load_kw=load_kw,
            pv_kw=pv_ac_kw,
            load_fc_kw=load_fc_kw,
            pv_fc_kw=pv_fc_kw,
            prev_dg_on=self.prev_dg_on
        )

        # Fuzzy action-command shaping
        decoded = self._fuzzy_action_shaping(decoded_raw, fuzzy_info, load_kw, pv_ac_kw)

        # Fuzzy penalty for critic learning through reward
        fuzzy_action_penalty = self.fuzzy.action_penalty(decoded, fuzzy_info, load_kw, pv_ac_kw)

        # Fuzzy-informed safety projection
        p_curt_ref_kw, pv_after_curt_kw, p_batt_cmd_ac_kw, p_diesel_safe_ref_kw = self._safety_projection(
            decoded,
            fuzzy_info,
            load_kw,
            pv_ac_kw
        )

        # Battery converter and battery model
        p_batt_internal_cmd, conv_loss_kw = self.converter.process_ac_to_internal(p_batt_cmd_ac_kw)
        p_batt_internal_actual, batt_deg_cost, soc, soh = self.battery.step(p_batt_internal_cmd, self.dt_h)
        p_batt_ac_kw = self.converter.internal_to_ac(p_batt_internal_actual)

        batt_charge_kw = max(p_batt_ac_kw, 0.0)
        batt_discharge_kw = max(-p_batt_ac_kw, 0.0)

        # Diesel-last after actual battery response
        supply_without_diesel_kw = pv_after_curt_kw - p_batt_ac_kw - conv_loss_kw
        residual_deficit_kw = max(load_kw - supply_without_diesel_kw, 0.0)

        optional_diesel_reserve_kw = 0.10 * p_diesel_safe_ref_kw if residual_deficit_kw > 1e-6 else 0.0

        # Fuzzy diesel permission is used here too.
        p_diesel_cmd_kw = residual_deficit_kw + optional_diesel_reserve_kw
        p_diesel_cmd_kw *= max(fuzzy_info["diesel_permission"], 0.05) if residual_deficit_kw > 1e-6 else 0.0

        # If there is still real unmet deficit, do not let fuzzy block reliability fully.
        if residual_deficit_kw > 1e-6 and soc <= self.battery.soc_min + 0.03:
            p_diesel_cmd_kw = max(p_diesel_cmd_kw, residual_deficit_kw)

        p_diesel_cmd_kw = float(np.clip(p_diesel_cmd_kw, 0.0, self.diesel.p_rated_kw))

        p_diesel_kw, fuel_l, dg_wear_cost, dg_status, dg_start = self.diesel.dispatch(
            p_diesel_cmd_kw,
            self.dt_h
        )

        # Final power balance
        supply_kw = pv_after_curt_kw + p_diesel_kw - p_batt_ac_kw - conv_loss_kw
        imbalance_kw = load_kw - supply_kw

        unmet_kw = max(imbalance_kw, 0.0)
        extra_curtail_kw = max(-imbalance_kw, 0.0)
        curtail_kw = p_curt_ref_kw + extra_curtail_kw

        served_kw = load_kw - unmet_kw

        line_loss_kw, i_line_A, v_bus_V, v_drop_V, current_violation = self.line.calc(served_kw)

        # Constraint checks
        soc_violation = max(self.battery.soc_min - soc, 0.0) + max(soc - self.battery.soc_max, 0.0)
        diesel_violation = max(p_diesel_kw - self.diesel.p_rated_kw, 0.0)
        batt_violation = max(abs(p_batt_ac_kw) - self.converter.p_rated_kw, 0.0)

        # Cost and reward
        c_fuel = fuel_l
        c_start = self.diesel.start_cost * dg_start
        c_wear = dg_wear_cost
        c_deg = batt_deg_cost
        c_curt = 2.00 * curtail_kw * self.dt_h
        c_unmet = 800.0 * unmet_kw * self.dt_h
        c_diesel_energy = 25.0 * p_diesel_kw * self.dt_h
        c_soc_tracking = 0.25 * abs(soc - decoded["soc_target"])

        c_constraint = (
            1000.0 * soc_violation +
            100.0 * float(current_violation) +
            100.0 * diesel_violation +
            100.0 * batt_violation
        )

        c_fuzzy = self.fuzzy_penalty_weight * fuzzy_action_penalty

        useful_discharge_reward = 5.0 * min(
            batt_discharge_kw,
            max(load_kw - pv_after_curt_kw, 0.0)
        ) * self.dt_h

        useful_charge_reward = 2.0 * min(
            batt_charge_kw,
            max(pv_after_curt_kw - load_kw, 0.0)
        ) * self.dt_h

        pv_used_kw = max(min(pv_after_curt_kw, load_kw + batt_charge_kw), 0.0)
        pv_utilization_reward = 2.0 * pv_used_kw * self.dt_h

        reward_raw = -(
            c_fuel +
            c_start +
            c_wear +
            c_deg +
            c_curt +
            c_unmet +
            c_diesel_energy +
            c_soc_tracking +
            c_constraint +
            c_fuzzy
        ) + useful_discharge_reward + useful_charge_reward + pv_utilization_reward

        reward = float(
            np.clip(
                reward_raw / max(self.reward_scale, 1e-9),
                self.reward_clip[0],
                self.reward_clip[1]
            )
        )

        self.records.append({
            "timestamp": ts,

            "load_kw": load_kw,
            "load_fc_kw": load_fc_kw,

            "pv_ac_kw": pv_ac_kw,
            "pv_dc_kw": pv_dc_kw,
            "pv_fc_kw": pv_fc_kw,
            "pv_after_curt_kw": pv_after_curt_kw,
            "net_load_kw": load_kw - pv_ac_kw,

            "raw_action_0": decoded_raw["raw_action_0"],
            "raw_action_1": decoded_raw["raw_action_1"],
            "raw_action_2": decoded_raw["raw_action_2"],
            "raw_action_3": decoded_raw["raw_action_3"],
            "raw_action_4": decoded_raw["raw_action_4"],

            "raw_batt_norm": decoded_raw["batt_norm"],
            "raw_diesel_frac": decoded_raw["diesel_frac"],
            "raw_curt_frac": decoded_raw["curt_frac"],
            "raw_soc_target": decoded_raw["soc_target"],
            "raw_dg_commit": decoded_raw["dg_commit"],
            "p_batt_ref_raw_kw": decoded_raw["p_batt_ref_kw"],
            "p_diesel_ref_raw_kw": decoded_raw["p_diesel_ref_kw"],

            "fuzzy_batt_norm": decoded["batt_norm"],
            "fuzzy_diesel_frac": decoded["diesel_frac"],
            "fuzzy_curt_frac": decoded["curt_frac"],
            "fuzzy_soc_target": decoded["soc_target"],
            "fuzzy_dg_commit": decoded["dg_commit"],
            "p_batt_ref_fuzzy_kw": decoded["p_batt_ref_kw"],
            "p_diesel_ref_fuzzy_kw": decoded["p_diesel_ref_kw"],

            "p_curt_ref_kw": p_curt_ref_kw,
            "p_batt_cmd_safe_kw": p_batt_cmd_ac_kw,
            "p_diesel_cmd_safe_kw": p_diesel_cmd_kw,

            "batt_ac_kw": p_batt_ac_kw,
            "batt_discharge_kw_positive": batt_discharge_kw,
            "batt_charge_kw_positive": batt_charge_kw,

            "diesel_kw": p_diesel_kw,
            "diesel_status": dg_status,
            "diesel_start": dg_start,
            "fuel_l": fuel_l,

            "dg_wear_cost": dg_wear_cost,
            "batt_deg_cost": batt_deg_cost,
            "conv_loss_kw": conv_loss_kw,
            "line_loss_kw": line_loss_kw,

            "curtail_kw": curtail_kw,
            "unmet_kw": unmet_kw,

            "i_line_A": i_line_A,
            "v_bus_V": v_bus_V,
            "v_drop_V": v_drop_V,
            "current_violation": current_violation,

            "soc": soc,
            "soh": soh,

            "soc_violation": soc_violation,
            "diesel_violation": diesel_violation,
            "batt_violation": batt_violation,

            "pitch_deg": pitch,
            "roll_deg": roll,
            "motion_loss": motion_loss,

            "fuzzy_balance_ratio": fuzzy_info["balance_ratio"],
            "fuzzy_forecast_balance_ratio": fuzzy_info["forecast_balance_ratio"],
            "fuzzy_charge_priority": fuzzy_info["charge_priority"],
            "fuzzy_discharge_priority": fuzzy_info["discharge_priority"],
            "fuzzy_diesel_permission": fuzzy_info["diesel_permission"],
            "fuzzy_curtail_permission": fuzzy_info["curtail_permission"],
            "fuzzy_soc_protection": fuzzy_info["soc_protection"],

            "mu_soc_critical": fuzzy_info["mu_soc_critical"],
            "mu_soc_low": fuzzy_info["mu_soc_low"],
            "mu_soc_medium": fuzzy_info["mu_soc_medium"],
            "mu_soc_high": fuzzy_info["mu_soc_high"],
            "mu_soc_full": fuzzy_info["mu_soc_full"],
            "mu_large_deficit": fuzzy_info["mu_large_deficit"],
            "mu_small_deficit": fuzzy_info["mu_small_deficit"],
            "mu_balanced": fuzzy_info["mu_balanced"],
            "mu_small_surplus": fuzzy_info["mu_small_surplus"],
            "mu_large_surplus": fuzzy_info["mu_large_surplus"],

            "fuzzy_action_penalty": fuzzy_action_penalty,
            "reward": reward,
            "reward_raw": reward_raw,

            "cost_fuel": c_fuel,
            "cost_start": c_start,
            "cost_wear": c_wear,
            "cost_deg": c_deg,
            "cost_curt": c_curt,
            "cost_unmet": c_unmet,
            "cost_diesel_energy": c_diesel_energy,
            "cost_soc_tracking": c_soc_tracking,
            "cost_constraint": c_constraint,
            "cost_fuzzy": c_fuzzy,

            "pv_used_kw": pv_used_kw,
            "pv_utilization_reward": pv_utilization_reward,
            "useful_discharge_reward": useful_discharge_reward,
            "useful_charge_reward": useful_charge_reward,
        })

        self.prev_batt_ac_kw = p_batt_ac_kw
        self.prev_diesel_kw = p_diesel_kw
        self.prev_dg_on = 1.0 if p_diesel_kw > 1e-6 else 0.0

        self.idx += 1
        self.steps += 1

        done = (
            self.steps >= self.episode_len or
            self.idx >= len(self.df) - self.forecast_horizon - 1
        )

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, done, {}

    def get_results(self):
        return pd.DataFrame(self.records).set_index("timestamp")


# ================================================================
# 4. SAC Networks and Replay Buffer
# ================================================================
class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))

        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = torch.tanh(mean)

        return action, log_prob, deterministic_action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=1))


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        target_entropy=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)

        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)

        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state_t)
            else:
                action, _, _ = self.policy.sample(state_t)

        return action.cpu().numpy()[0]

    def train_step(self, replay, batch_size=256):
        if len(replay) < batch_size:
            return None

        s, a, r, s2, d = replay.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(s2)
            q_next = torch.min(
                self.q1_target(s2, a2),
                self.q2_target(s2, a2)
            ) - self.alpha.detach() * logp2

            y = r + self.gamma * (1.0 - d) * q_next

        q1_loss = F.mse_loss(self.q1(s, a), y)
        q2_loss = F.mse_loss(self.q2(s, a), y)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        new_a, logp, _ = self.policy.sample(s)
        q_new = torch.min(self.q1(s, new_a), self.q2(s, new_a))

        policy_loss = (self.alpha.detach() * logp - q_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update(self.q1_target, self.q1)
        self.soft_update(self.q2_target, self.q2)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item())
        }

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, folder="sac_microgrid_model_fuzzy_embedded"):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{folder}/policy.pth")
        torch.save(self.q1.state_dict(), f"{folder}/q1.pth")
        torch.save(self.q2.state_dict(), f"{folder}/q2.pth")
        torch.save(self.log_alpha.detach().cpu(), f"{folder}/log_alpha.pt")


# ================================================================
# 5. Training and Evaluation
# ================================================================
def evaluate_policy_reward(env, agent):
    state = env.reset(random_start=False)
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    return float(total_reward)


def train_sac(
    env,
    episodes=800,
    batch_size=256,
    warmup_steps=3000,
    updates_per_step=1,
    validation_every=20
):
    agent = SACAgent(env.state_dim, env.action_dim)
    replay = ReplayBuffer()

    rewards = []
    moving_rewards = []
    validation_rewards = []

    loss_history = {
        "q1_loss": [],
        "q2_loss": [],
        "policy_loss": [],
        "alpha_loss": [],
        "alpha": []
    }

    total_steps = 0

    for ep in range(episodes):
        state = env.reset(random_start=True)
        done = False
        ep_reward = 0.0

        while not done:
            if total_steps < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, _ = env.step(action)

            replay.push(state, action, reward, next_state, float(done))

            if total_steps >= warmup_steps:
                for _ in range(updates_per_step):
                    info = agent.train_step(replay, batch_size=batch_size)
                    if info is not None:
                        for k in loss_history:
                            loss_history[k].append(info[k])

            state = next_state
            ep_reward += reward
            total_steps += 1

        rewards.append(ep_reward)
        moving_rewards.append(float(pd.Series(rewards).rolling(20, min_periods=1).mean().iloc[-1]))

        if (ep + 1) % validation_every == 0:
            val_reward = evaluate_policy_reward(env, agent)
            validation_rewards.append({
                "episode": ep + 1,
                "validation_reward": val_reward
            })

            alpha_val = loss_history["alpha"][-1] if loss_history["alpha"] else np.nan

            print(
                f"Episode {ep+1}/{episodes} | "
                f"Reward={ep_reward:.3f} | "
                f"MA20={moving_rewards[-1]:.3f} | "
                f"Val={val_reward:.3f} | "
                f"Replay={len(replay)} | "
                f"Alpha={alpha_val:.4f}"
            )

        elif (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Reward={ep_reward:.3f} | "
                f"MA20={moving_rewards[-1]:.3f} | "
                f"Replay={len(replay)}"
            )

    agent.save()

    return agent, rewards, moving_rewards, validation_rewards, loss_history


def evaluate_agent(env, agent):
    state = env.reset(random_start=False)
    done = False

    while not done:
        action = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)

    return env.get_results()


# ================================================================
# 6. KPI Calculation
# ================================================================
def compute_kpis(results, dt_h=0.25, fuel_cost_per_l=1.0, co2_factor=2.68):
    E = lambda col: (results[col] * dt_h).sum() if col in results.columns else 0.0

    load_kwh = E("load_kw")
    unmet_kwh = E("unmet_kw")
    served_kwh = load_kwh - unmet_kwh

    fuel_l = results["fuel_l"].sum()
    fuel_cost = fuel_l * fuel_cost_per_l

    dg_wear_cost = results["dg_wear_cost"].sum()
    batt_deg_cost = results["batt_deg_cost"].sum()

    total_cost = fuel_cost + dg_wear_cost + batt_deg_cost

    return pd.DataFrame([{
        "Load_kWh": load_kwh,
        "Served_kWh": served_kwh,
        "Unmet_kWh": unmet_kwh,
        "LPSP_%": 100 * unmet_kwh / max(load_kwh, 1e-9),

        "PV_available_kWh": E("pv_ac_kw"),
        "PV_after_curtail_kWh": E("pv_after_curt_kw"),
        "Diesel_kWh": E("diesel_kw"),
        "Curtail_kWh": E("curtail_kw"),

        "Fuel_L": fuel_l,
        "CO2_kg": fuel_l * co2_factor,
        "CO2_ton": fuel_l * co2_factor / 1000,

        "Batt_charge_kWh": E("batt_charge_kw_positive"),
        "Batt_discharge_kWh": E("batt_discharge_kw_positive"),

        "Line_loss_kWh": E("line_loss_kw"),
        "Converter_loss_kWh": E("conv_loss_kw"),

        "Fuel_cost_$": fuel_cost,
        "DG_wear_cost_$": dg_wear_cost,
        "Battery_deg_cost_$": batt_deg_cost,
        "Total_cost_$": total_cost,
        "Cost_per_kWh_served_$": total_cost / max(served_kwh, 1e-9),

        "Renewable_fraction_%": 100 * (
            E("pv_after_curt_kw") + E("batt_discharge_kw_positive")
        ) / max(load_kwh, 1e-9),

        "Diesel_fraction_%": 100 * E("diesel_kw") / max(load_kwh, 1e-9),

        "SOC_min": results["soc"].min(),
        "SOC_mean": results["soc"].mean(),
        "SOC_max": results["soc"].max(),
        "SOH_final": results["soh"].iloc[-1],

        "Diesel_starts": int(results["diesel_start"].sum()),
        "Current_violations": int(results["current_violation"].sum()),

        "SOC_viol_sum": results["soc_violation"].sum(),
        "Diesel_viol_sum": results["diesel_violation"].sum(),
        "Battery_viol_sum": results["batt_violation"].sum(),

        "Voltage_min_V": results["v_bus_V"].min(),
        "Voltage_mean_V": results["v_bus_V"].mean(),
        "Voltage_max_V": results["v_bus_V"].max(),

        "Fuzzy_action_penalty_sum": results["fuzzy_action_penalty"].sum(),
        "Fuzzy_cost_sum": results["cost_fuzzy"].sum(),

        "Avg_fuzzy_charge_priority": results["fuzzy_charge_priority"].mean(),
        "Avg_fuzzy_discharge_priority": results["fuzzy_discharge_priority"].mean(),
        "Avg_fuzzy_diesel_permission": results["fuzzy_diesel_permission"].mean(),
        "Avg_fuzzy_curtail_permission": results["fuzzy_curtail_permission"].mean(),
        "Avg_fuzzy_soc_protection": results["fuzzy_soc_protection"].mean(),

        "Total_reward": results["reward"].sum(),
        "Total_raw_reward": results["reward_raw"].sum(),
    }])


# ================================================================
# 7. Plotting
# ================================================================
def plot_results(
    results,
    rewards,
    moving_rewards=None,
    validation_rewards=None,
    loss_history=None,
    save_dir="plots_fuzzy_embedded_sac"
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
    plt.savefig(f"{save_dir}/training_reward_sac.png", dpi=300)
    plt.close()

    if loss_history is not None:
        for key, ylabel in [
            ("q1_loss", "Q1 Loss"),
            ("q2_loss", "Q2 Loss"),
            ("policy_loss", "Policy Loss"),
            ("alpha", "Entropy Temperature Alpha")
        ]:
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
    plt.savefig(f"{save_dir}/power_profile_fuzzy_embedded_sac.png", dpi=300)
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
    plt.savefig(f"{save_dir}/battery_raw_fuzzy_safe_actual.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["p_diesel_ref_raw_kw"], label="Raw SAC Diesel Ref")
    plt.plot(results.index, results["p_diesel_ref_fuzzy_kw"], label="Fuzzy-Shaped Diesel Ref")
    plt.plot(results.index, results["p_diesel_cmd_safe_kw"], label="Safe Diesel Command")
    plt.plot(results.index, results["diesel_kw"], label="Actual Diesel")
    plt.ylabel("Diesel Power (kW)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/diesel_raw_fuzzy_safe_actual.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["soc"], label="SOC")
    plt.plot(results.index, results["soh"], label="SOH")
    plt.plot(results.index, results["raw_soc_target"], label="Raw SAC SOC Target")
    plt.plot(results.index, results["fuzzy_soc_target"], label="Fuzzy-Shaped SOC Target")
    plt.ylabel("SOC / SOH")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/soc_soh_target.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["fuzzy_charge_priority"], label="Charge Priority")
    plt.plot(results.index, results["fuzzy_discharge_priority"], label="Discharge Priority")
    plt.plot(results.index, results["fuzzy_diesel_permission"], label="Diesel Permission")
    plt.plot(results.index, results["fuzzy_curtail_permission"], label="Curtail Permission")
    plt.plot(results.index, results["fuzzy_soc_protection"], label="SOC Protection")
    plt.ylabel("Fuzzy Output")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fuzzy_outputs.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(results.index, results["curtail_kw"], label="PV Curtailment")
    plt.plot(results.index, results["unmet_kw"], label="Unmet Load")
    plt.plot(results.index, results["fuzzy_action_penalty"], label="Fuzzy Action Penalty")
    plt.ylabel("Power / Penalty")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/curtailment_unmet_fuzzy_penalty.png", dpi=300)
    plt.close()

    print(f"Plots saved in: {save_dir}")


# ================================================================
# 8. Main Execution
# ================================================================
if __name__ == "__main__":
    set_seed(42)

    controller_name = "fuzzy_embedded_safety_constrained_sac"
    dt_h = 15 / 60

    df_train = load_training_data_from_load_with_weather(
        training_file="load_with_weather.csv",
        start_date="2016-01-01",
        end_date="2018-12-31 23:59:59"
    )

    df_test = load_test_operation_data(
        test_file="target_load.csv"
    )

    train_env = FuzzyEmbeddedSACMicrogridEnv(
        df_train,
        dt_h=dt_h,
        episode_len=96,
        forecast_horizon=4,
        seed=42,
        reward_scale=100.0,
        fuzzy_action_gain=0.35,
        fuzzy_penalty_weight=20.0
    )

    test_env = FuzzyEmbeddedSACMicrogridEnv(
        df_test,
        dt_h=dt_h,
        episode_len=max(len(df_test) - 5, 1),
        forecast_horizon=4,
        seed=99,
        reward_scale=100.0,
        fuzzy_action_gain=0.35,
        fuzzy_penalty_weight=20.0
    )

    agent, rewards, moving_rewards, validation_rewards, loss_history = train_sac(
        env=train_env,
        episodes=800,
        batch_size=256,
        warmup_steps=3000,
        updates_per_step=1,
        validation_every=20
    )

    results = evaluate_agent(test_env, agent)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = f"results_{controller_name}_{timestamp_tag}.csv"
    kpi_file = f"kpis_{controller_name}_{timestamp_tag}.csv"
    reward_file = f"training_rewards_{controller_name}_{timestamp_tag}.csv"
    validation_file = f"validation_rewards_{controller_name}_{timestamp_tag}.csv"
    loss_file = f"training_losses_{controller_name}_{timestamp_tag}.csv"
    plot_dir = f"plots_{controller_name}_{timestamp_tag}"

    results.to_csv(results_file)

    pd.DataFrame({
        "episode_reward": rewards,
        "episode_reward_MA20": moving_rewards
    }).to_csv(reward_file, index=False)

    pd.DataFrame(validation_rewards).to_csv(validation_file, index=False)

    pd.DataFrame({
        key: pd.Series(value) for key, value in loss_history.items()
    }).to_csv(loss_file, index=False)

    kpis = compute_kpis(
        results,
        dt_h=dt_h,
        fuel_cost_per_l=1.0
    )

    kpis.to_csv(kpi_file, index=False)

    print("\n===== KPI SUMMARY ON TEST DATA =====")
    print(kpis.T)

    print("\n===== ACTION COMMAND CHECK =====")
    print(results[[
        "p_batt_ref_raw_kw",
        "p_batt_ref_fuzzy_kw",
        "p_batt_cmd_safe_kw",
        "batt_ac_kw",
        "p_diesel_ref_raw_kw",
        "p_diesel_ref_fuzzy_kw",
        "p_diesel_cmd_safe_kw",
        "diesel_kw",
        "raw_soc_target",
        "fuzzy_soc_target",
        "soc",
        "unmet_kw",
        "curtail_kw",
        "fuzzy_action_penalty"
    ]].describe())

    print("\n===== FUZZY OUTPUT CHECK =====")
    print(results[[
        "fuzzy_charge_priority",
        "fuzzy_discharge_priority",
        "fuzzy_diesel_permission",
        "fuzzy_curtail_permission",
        "fuzzy_soc_protection"
    ]].describe())

    print("\n===== CONSTRAINT VIOLATION CHECK =====")
    print("SOC violation sum    :", results["soc_violation"].sum())
    print("Diesel violation sum :", results["diesel_violation"].sum())
    print("Battery violation sum:", results["batt_violation"].sum())
    print("Current violations   :", int(results["current_violation"].sum()))

    print(f"\nResults saved to            : {results_file}")
    print(f"KPIs saved to               : {kpi_file}")
    print(f"Training rewards saved to   : {reward_file}")
    print(f"Validation rewards saved to : {validation_file}")
    print(f"Training losses saved to    : {loss_file}")

    plot_results(
        results,
        rewards,
        moving_rewards,
        validation_rewards,
        loss_history,
        save_dir=plot_dir
    )

    print("\nFuzzy-Embedded Safety-Constrained Hierarchical SAC training and testing complete.")
