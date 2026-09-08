import numpy as np

class FuzzyEmbeddedSupervisor:
    """
    Supervisory fuzzy logic layer for EMS optimization.
    Generates priorities and permissions to constrain and guide RL actions.
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

        if prev_dg_on > 0.5:
            diesel_rules.append((mu_bal["small_deficit"], 0.35))
            diesel_rules.append((mu_bal["large_deficit"], 0.60))

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

        soc_protect_rules = [
            (mu_soc["critical"], 1.00),
            (mu_soc["low"], 0.65),
            (mu_soc["medium"], 0.15),
            (mu_soc["high"], 0.35),
            (mu_soc["full"], 1.00),
        ]

        return {
            "balance_ratio": float(balance_ratio),
            "forecast_balance_ratio": float(forecast_ratio),
            "charge_priority": float(np.clip(self.weighted_average(charge_rules, 0.0), 0.0, 1.0)),
            "discharge_priority": float(np.clip(self.weighted_average(discharge_rules, 0.0), 0.0, 1.0)),
            "diesel_permission": float(np.clip(self.weighted_average(diesel_rules, 0.0), 0.0, 1.0)),
            "curtail_permission": float(np.clip(self.weighted_average(curtail_rules, 0.0), 0.0, 1.0)),
            "soc_protection": float(np.clip(self.weighted_average(soc_protect_rules, 0.2), 0.0, 1.0)),
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
        deficit_kw = max(load_kw - pv_kw, 0.0)
        surplus_kw = max(pv_kw - load_kw, 0.0)

        p_batt_ref = decoded["p_batt_ref_kw"]
        p_diesel_ref = decoded["p_diesel_ref_kw"]
        curt_frac = decoded["curt_frac"]

        penalty = 0.0

        if p_batt_ref < 0:
            penalty += fuzzy["soc_protection"] * (1.0 - fuzzy["discharge_priority"]) * abs(p_batt_ref) / 500.0

        if p_batt_ref > 0 and fuzzy["charge_priority"] < 0.15:
            penalty += (0.15 - fuzzy["charge_priority"]) * p_batt_ref / 500.0

        if p_diesel_ref > 1e-6:
            if deficit_kw <= 1e-6:
                penalty += 0.75 * p_diesel_ref / 120.0
            penalty += (1.0 - fuzzy["diesel_permission"]) * p_diesel_ref / 120.0

        if surplus_kw > 1e-6 and curt_frac > 0:
            penalty += (1.0 - fuzzy["curtail_permission"]) * curt_frac

        return float(max(penalty, 0.0))
