import random
import numpy as np
import pandas as pd
import torch

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

def load_data(file_path, start_date=None, end_date=None):
    df = pd.read_csv(file_path)

    ts_col = detect_column(df, ["timestamp", "Timestamp", "DateTime", "Datetime", "date_time", "Date Time", "Time", "Date"], "timestamp")
    load_col = detect_column(df, ["load_kw", "Load_kW", "Load", "load", "Power_kW", "RealPower", "power_kw", "kW", "KW"], "load")
    ghi_col = detect_column(df, ["ghi_wm2", "GHI_Wm2", "GHI", "Irradiance", "irradiance", "ALLSKY_SFC_SW_DWN"], "irradiance")

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["load_kw"] = pd.to_numeric(df[load_col], errors="coerce").clip(lower=0)
    df["ghi_wm2"] = pd.to_numeric(df[ghi_col], errors="coerce").clip(lower=0)

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

    if start_date and end_date:
        df = df[(df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date))]

    df = df[["timestamp", "load_kw", "ghi_wm2", "cell_temp_c"]].sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"Dataframe from {file_path} is empty.")

    print(f"Loaded {len(df)} samples from {file_path}. Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

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
        "Renewable_fraction_%": 100 * (E("pv_after_curt_kw") + E("batt_discharge_kw_positive")) / max(load_kwh, 1e-9),
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
