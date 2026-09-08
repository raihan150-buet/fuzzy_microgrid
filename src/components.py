import numpy as np
from dataclasses import dataclass, field

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
