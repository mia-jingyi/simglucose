import copy

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatientNew
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import (SemiRandomBalancedScenario, RandomBalancedScenario,
                                                CustomBalancedScenario)
from simglucose.controller.base import Action
from simglucose.controller.pid_ctrller import PIDController
from simglucose.analysis.risk import magni_risk_index

import pandas as pd
import numpy as np
import joblib
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime


class DeepSACT1DEnv(gym.Env):
    """A gym environment supporting SAC learning. Uses PID control for initialization

    Args:
        reward_fun: reward function.
        patient_name: string, name of a specific patient, default is adolescent#001.
        seeds: dictionary, storing seeds for numpy, sensor, scenario and patient.
        n_hours: integer, window of observation to provide temporal context.
        norm: boolean. If true, state is normalized.
        time: boolean. If true, add history of sin/cos normalized time to the state.
        weekly: boolean. If true, create weekend scenario for weekends and include binary weekend flag in the state.
        meal: boolean. If true, add history of carbohydrate intake to the state.
        fake_real: boolean. If true, state is an unrolled vector of repeated ground truth states.
        gt: boolean. If true, state is a 1D vector consisting of groung truth values.
        fake_gt: boolean. If true, state = [cgm, iob]; otherwise  state is the ground truth 13D patient state vector.
        suppress_carbs: boolean. If true, set x0, x1, and x2 to 0 (empty glucose in stomach and intestine).
        limited_gt: boolean. If true, state consists of plasma glucose and iob.
        action_cap: integer, the upper limit of action space.
        action_bias: integer, bias of the insulin dose, probably related to the noise of insulin pump.
        action_scale: "basal" or integer. If "basal",  scale the action such that the maximum amount of insulin
                      delivered over a 5-minute interval is roughly equal to a normal meal bolus for each individual;
                      otherwise it's a scaling factor by which the action is multiplied.
        basal_scaling: integer, a scale parameter used when action_scale is "basal".
        meal_announce: None or integer. If integer, future meal amount and meal time are added to the state.
        residual_basal: boolean. If true, add patient_specific basal insulin rate to the action.
        residual_bolus: boolean. If true, add meal bolus to the action when there is an upcoming meal.
        carb_miss_prob: integer, indicating the deviation of the actual carb intake from the meal amount
                        recorded in the meal schedule.
        carb_error_std: integer, indicating the probability that the meal appeared in the meal schedule is skipped.
        residual_PID: boolean. If true, add insulin rate determined by the PID controller to the action.
        rolling_insulin_lim: None or integer. If integer, it defines the maximum sum of insulin dose accumulated from
                             previous 12 steps till the current step.
        reset_lim: dictionary, specifying the lower and upper limits for ending the episode.
        termination_penalty: None or integer to avoid early termination.
        reward_bias: integer, contributing to the actual reward.
        load: boolean. If true, load the environment from PKL file.
        use_old_patient_env: boolean. If true, environment with the old patient class is loaded.
        use_model: boolean. If true, an external model is used to predict the next state given the current ground truth
                   13D patient state vector, CHO and insulin.
        model: external model for predicting the next state.
        model_device: cpu or gpu if available.
        use_pid_load: boolean. If true, state is initialized by controlling bg with pid contorller for 1 day.
        hist_init: boolean. If true, state is initialized by loading {patient_name}_data.pkl.
        start_date: simulation start date. Simulation starts at 2022/01/01 0:00 if start_date is None.
        time_std: None or integer. If integer, it determines the std of meal time when generating semi-random
                  balanced scenario.
        harrison_benedict: boolean. If true, use the harrison_benedict meal schedule.
        restricted_carb: boolean. If true, create restricted scenario defined in RandomBalancedScenario.
        meal_duration: integer, specifying the meal duration in RandomBalancedScenario and SemiRandomBalancedScenario.
        unrealistic:boolean. If true, create unrealistic scenario defined in RandomBalancedScenario.
        deterministic_meal_size: boolean. If true, meal size is deterministic in the normal scenario defined in
                                 RandomBalancedScenario.
        deterministic_meal_time: boolean. If true, meal time is deterministic in the normal scenario defined in
                                 RandomBalancedScenario.
        deterministic_meal_occurrence: boolean. If true, meal occurrence is deterministic in the normal scenario defined
                                 in RandomBalancedScenario.
        use_custom_meal: boolean. If true, a custom balanced scenario is generated.
        custom_meal_num: integer, specifying the number of custom meals when use_custom_meal is True.
        custom_meal_size: integer, specifying the size of custom meals when use_custom_meal is True.
        update_seed_on_reset: boolean. If true, seeds are updated whenever the env is reset.
        source_dir: the path to the location of the folder 'simglucose' which contains the source code.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_fun, patient_name=None, seeds=None, n_hours=24, norm=False, time=False,
                 weekly=False, meal=False, fake_real=False, gt=False, fake_gt=False, suppress_carbs=False,
                 limited_gt=False, action_cap=0.1, action_bias=0, action_scale=1, basal_scaling=43.2,
                 meal_announce=None, residual_basal=False, residual_bolus=False,
                 carb_miss_prob=0, carb_error_std=0, residual_PID=False, rolling_insulin_lim=None, reset_lim=None,
                 termination_penalty=None, reward_bias=0, load=False, use_old_patient_env=False,
                 use_model=False, model=None, model_device='cpu', use_pid_load=False, hist_init=False,
                 start_date=None, time_std=None, harrison_benedict=False, restricted_carb=False, meal_duration=1,
                 unrealistic=False, deterministic_meal_size=False, deterministic_meal_time=False,
                 deterministic_meal_occurrence=False, use_custom_meal=False, custom_meal_num=3, custom_meal_size=1,
                 update_seed_on_reset=False, source_dir=None, **kwargs):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        self.source_dir = source_dir
        self.patient_para_file = '{}/simglucose/params/vpatient_params.csv'.format(self.source_dir)
        self.control_quest = '{}/simglucose/params/Quest2.csv'.format(self.source_dir)
        self.pid_para_file = '{}/simglucose/params/pid_params.csv'.format(self.source_dir)
        self.pid_env_path = '{}/simglucose/params'.format(self.source_dir)
        self.sensor_para_file = '{}/simglucose/params/sensor_params.csv'.format(self.source_dir)
        self.insulin_pump_para_file = '{}/simglucose/params/pump_params.csv'.format(self.source_dir)
        if patient_name is None:
            patient_name = 'adolescent#001'
        if seeds is None:
            seed_list = self._seed()
            seeds = {'numpy': seed_list[0], 'sensor': seed_list[1], 'scenario': seed_list[2], 'patient': seed_list[3]}
        self.seeds = seeds
        self.sample_time = 5
        self.day = int(1440 / self.sample_time)  # number of samples per day
        self.state_hist = int((n_hours * 60) / self.sample_time)  # number of samples in the observation space
        self.norm = norm
        self.time = time
        self.weekly = weekly
        self.meal = meal
        self.fake_real = fake_real
        self.gt = gt
        self.fake_gt = fake_gt
        self.suppress_carbs = suppress_carbs
        self.limited_gt = limited_gt
        self.reward_fun = reward_fun
        self.action_cap = action_cap
        self.action_bias = action_bias
        self.action_scale = action_scale
        self.basal_scaling = basal_scaling
        self.meal_announce = meal_announce
        self.residual_basal = residual_basal
        self.residual_bolus = residual_bolus
        self.carb_miss_prob = carb_miss_prob
        self.carb_error_std = carb_error_std
        self.residual_PID = residual_PID
        self.rolling_insulin_lim = rolling_insulin_lim
        self.rolling = []
        if reset_lim is None:
            self.reset_lim = {'lower_lim': 10, 'upper_lim': 1000}
        else:
            self.reset_lim = reset_lim
        self.termination_penalty = termination_penalty
        self.reward_bias = reward_bias
        self.target = 140
        self.low_lim = 140  # Matching BB controller
        self.cooldown = 180  # corresponding to 3 hrs
        self.last_cf = self.cooldown + 1
        self.load = load
        self.use_old_patient_env = use_old_patient_env
        self.use_model = use_model
        self.model = model
        self.model_device = model_device
        self.use_pid_load = use_pid_load
        self.hist_init = hist_init
        self.start_date = start_date
        if self.start_date is None:
            start_time = datetime(2022, 1, 1, 0, 0, 0)
        else:
            start_time = datetime(self.start_date.year, self.start_date.month, self.start_date.day, 0, 0, 0)
        self.start_time = start_time
        self.time_std = time_std
        self.harrison_benedict = harrison_benedict
        self.restricted_carb = restricted_carb
        self.meal_duration = meal_duration
        self.unrealistic = unrealistic
        self.deterministic_meal_size = deterministic_meal_size
        self.deterministic_meal_time = deterministic_meal_time
        self.deterministic_meal_occurrence = deterministic_meal_occurrence
        self.use_custom_meal = use_custom_meal
        self.custom_meal_num = custom_meal_num
        self.custom_meal_size = custom_meal_size
        self.update_seed_on_reset = update_seed_on_reset
        self.set_patient_dependent_values(patient_name)
        self.env.scenario.day = 0

    def pid_load(self, n_days):
        """
        Control BG using PID controller for n_days.
        """
        for i in range(n_days*self.day):
            b_val = self.pid.step(self.env.GCM.hist[-1])
            act = Action(basal=0, bolus=b_val)
            _ = self.env.step(action=act, reward_fun=self.reward_fun)

    def step(self, action):
        return self._step(action)

    def _step(self, action, use_action_scale=True):
        if use_action_scale:
            action = self.translate(action)
        if self.residual_basal:
            action += self.ideal_basal
        if self.residual_bolus:
            ma = self.announce_meal(5)
            carbs = ma[0]
            if np.random.uniform() < self.carb_miss_prob:
                carbs = 0
            error = np.random.normal(0, self.carb_error_std)
            carbs = carbs + carbs * error
            glucose = self.env.CGM_hist[-1]
            if carbs > 0:
                carb_correct = carbs / self.CR
                hyper_correct = (glucose > self.target) * (glucose - self.target) / self.CF
                hypo_correct = (glucose < self.low_lim) * (self.low_lim - glucose) / self.CF
                bolus = 0
                # if there have been no meals in the past three hours
                if self.last_cf > self.cooldown:
                    bolus += hyper_correct - hypo_correct
                bolus += carb_correct
                action += bolus / 5  # bolus per min
                self.last_cf = 0
            self.last_cf += 5
        if self.residual_PID:
            action += self.pid.step(self.env.CGM_hist[-1])
        if self.action_cap is not None:
            action = min(self.action_cap, action)
        if self.rolling_insulin_lim is not None:
            if np.sum(self.rolling + [action]) > self.rolling_insulin_lim:
                action = max(0, action - (np.sum(self.rolling + [action]) - self.rolling_insulin_lim))
            self.rolling.append(action)
            if len(self.rolling) > 12:
                self.rolling = self.rolling[1:]
        act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun)
        state = self.get_state(self.norm)
        done = self.is_done()
        if done and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        reward = reward + self.reward_bias
        return state, reward, done, info

    def translate(self, action):
        """Scale the action space"""
        if self.action_scale == 'basal':
            # 288 samples per day, bolus insulin should be 75% of insulin dose
            # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
            # https://care.diabetesjournals.org/content/34/5/1089
            action = (action + self.action_bias) * ((self.ideal_basal * self.basal_scaling) / (1 + self.action_bias))
        else:
            action = (action + self.action_bias) * self.action_scale
        return max(0, action)

    def announce_meal(self, meal_announce=None):
        """Check whether there is upcoming meal given the current time
        args:
            meal_announce: integer that defines how many minutes in advance the meal is announce.
        returns:
            meal amount, meal time
        """
        t = self.env.time.hour * 60 + self.env.time.minute  # Assuming 5 minute sampling rate
        for i, m_t in enumerate(self.env.scenario.scenario['meal']['time']):
            # round up to nearst 5
            if m_t % 5 != 0:
                m_tr = m_t - (m_t % 5) + 5
            else:
                m_tr = m_t
            if meal_announce is None:
                ma = self.meal_announce
            else:
                ma = meal_announce
            if t < m_tr <= t + ma:
                return self.env.scenario.scenario['meal']['amount'][i], m_tr - t
        return 0, 0

    def calculate_iob(self):
        """Calculate the total amount of insulin on board (IOB), which refers to insulin that has been
        infused but still working on the body."""
        ins = self.env.insulin_hist
        return np.dot(np.flip(self.iob, axis=0)[-len(ins):], ins[-len(self.iob):])

    def get_state(self, normalize=False):
        bg = self.env.CGM_hist[-self.state_hist:]
        insulin = self.env.insulin_hist[-self.state_hist:]
        if normalize:
            # max scaling because we set the maximum value that we can have
            bg = np.array(bg) / 400.
            insulin = np.array(insulin) * 10
        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))  # pad with -1
        if len(insulin) < self.state_hist:
            insulin = np.concatenate((np.full(self.state_hist - len(insulin), -1), insulin))  # pad with -1
        return_arr = [bg, insulin]
        if self.time:
            time_dt = self.env.time_hist[-self.state_hist:]
            time = np.array([(t.minute + 60 * t.hour) / self.sample_time for t in time_dt])
            sin_time = np.sin(time * 2 * np.pi / self.day)
            cos_time = np.cos(time * 2 * np.pi / self.day)
            if normalize:
                pass  # already normalized
            if len(sin_time) < self.state_hist:
                sin_time = np.concatenate((np.full(self.state_hist - len(sin_time) - 1), sin_time))
            if len(cos_time) < self.state_hist:
                cos_time = np.concatenate((np.full(self.state_hist - len(cos_time), -1), cos_time))
            return_arr.append(sin_time)
            return_arr.append(cos_time)
            if self.weekly:
                # binary flag signalling weekend
                # there might be a bug: what if we have one weekend day and one workday
                if self.env.scenario.day == 5 or self.env.scenario.day == 6:
                    return_arr.append(np.full(self.state_hist, 1))  # append the state with 1 if weekend
                else:
                    return_arr.append(np.full(self.state_hist, 0))  # append the state with 0 if not weekend
        if self.meal:
            cho = self.env.CHO_hist[-self.state_hist:]
            if normalize:
                cho = np.array(cho) / 20.
            if len(cho) < self.state_hist:
                cho = np.concatenate((np.full(self.state_hist - len(cho), -1), cho))
            return_arr.append(cho)
        if self.meal_announce is not None:
            # update every {sample_time} minutes
            meal_val, meal_time = self.announce_meal()
            future_cho = np.full(self.state_hist, meal_val)
            return_arr.append(future_cho)
            future_time = np.full(self.state_hist, meal_time)
            return_arr.append(future_time)
        if self.fake_real:
            state = self.env.patient.state
            return np.stack([state for _ in range(self.state_hist)]).T.flatten()
        if self.gt:
            if self.fake_gt:
                iob = self.calculate_iob()
                cgm = self.env.CGM_hist[-1]
                if normalize:
                    state = np.array([cgm/400, iob*10])
                else:
                    state = np.array([cgm, iob])
            else:
                state = self.env.patient.state
            if self.meal_announce is not None:
                meal_val, meal_time = self.announce_meal()
                state = np.array((state, np.array([meal_val, meal_time])))
            if normalize:
                # just the average of 2 days of adult#001, these values are patient-specific
                # every patient has different value ranges
                # not sure how this array was generated
                norm_arr = np.array([4.86688301e+03, 4.95825609e+03, 2.52219425e+03, 2.73376341e+02,
                                     1.56207049e+02, 9.72051746e+00, 7.65293763e+01, 1.76808549e+02,
                                     1.76634852e+02, 5.66410518e+00, 1.28448645e+02, 2.49195394e+02,
                                     2.73250649e+02, 7.70883882e+00, 1.63778163e+00])
                if self.meal_announce is not None:
                    state = state/norm_arr
                else:
                    state = state/norm_arr[:-2]
            if self.suppress_carbs:
                # state[0] = stomach solid
                # state[1] = stomach liquid
                # state[2] = gut
                state[:3] = 0.
            if self.limited_gt:
                # state[3] = plasma glucose
                state = np.array([state[3], self.calculate_iob()])
            return state
        return np.stack(return_arr).flatten()

    def avg_risk(self):
        # ignore the first 288 samples because those samples were initialized with pid or exsiting pkl file
        # both initialization method generates 288 samples (1 day)
        return np.mean(self.env.risk_hist[max(self.state_hist, 288):])

    def avg_magni_risk(self):
        return np.mean(self.env.magni_risk_hist[max(self.state_hist, 288):])

    def glycemic_report(self):
        bg = np.array(self.env.BG_hist[max(self.state_hist, 288):])
        ins = np.array(self.env.insulin_hist[max(self.state_hist, 288):])
        hypo = (bg < 70).sum()/len(bg)
        hyper = (bg > 180).sum()/len(bg)
        euglycemic = 1 - (hypo + hyper)
        return bg, euglycemic, hypo, hyper, ins

    def is_done(self):
        return self.env.BG_hist[-1] < self.reset_lim['lower_lim'] or self.env.BG_hist[-1] > self.reset_lim['upper_lim']

    def increment_seed(self, incr=1):
        self.seeds['numpy'] += incr
        self.seeds['sensor'] += incr
        self.seeds['scenario'] += incr
        self.seeds['patient'] += incr

    def reset(self):
        return self._reset()

    def _reset(self):
        if self.update_seed_on_reset:
            self.increment_seed()
        if self.use_model:
            if self.load:
                self.env = joblib.load("{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name))
                self.env.model = self.model
                self.env.model_device = self.model_device
                self.env.state = self.env.patient.state
                self.env.scenario.kind = self.kind
            else:
                self.env.reset()
        else:
            # We need a functionality of saving the environment in the future.
            if self.load:
                if self.use_old_patient_env:
                    self.env = joblib.load("{}/{}_env.pkl".format(self.pid_env_path, self.patient_name))
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                else:
                    self.env = joblib.load('{}/{}_fenv.pkl'.format(self.pid_env_path, self.patient_name))
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                if self.time_std is not None:
                    self.env.scenario = SemiRandomBalancedScenario(bw=self.bw, start_time=self.start_time,
                                                                   seed=self.seeds['scenario'],
                                                                   time_std_multiplier=self.time_std, kind=self.kind,
                                                                   harrison_benedict=self.harrison_benedict,
                                                                   meal_duration=self.meal_duration)
                self.env.sensor.seed = self.seeds['sensor']
                self.env.scenario.seed = self.seeds['scenario']
                self.env.scenario.day = 0
                self.env.scenario.weekly = self.weekly
                self.env.scenario.kind = self.kind
                self.env.paitent.seed = self.seeds['patient']
            else:
                self.env.sensor.seed = self.seeds['sensor']
                self.env.scenario.seed = self.seeds['scenario']
                self.env.patient.seed = self.seeds['patient']
                self.env.reset()
                self.pid.reset()
                if self.use_pid_load:
                    self.pid_load(1)
                if self.hist_init:
                    self._hist_init()
        return self.get_state(self.norm)

    def set_patient_dependent_values(self, patient_name):
        self.patient_name = patient_name
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split('#')[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))['BW'].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))['u2ss'].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        if self.rolling_insulin_lim is not None:
            # not sure how this equation is derived
            self.rolling_insulin_lim = ((self.rolling_insulin_lim * self.bw) / self.CR * self.rolling_insulin_lim) / 5
        else:
            self.rolling_insulin_lim = None
        iob_all = joblib.load('{}/iob.pkl'.format(self.pid_env_path))
        self.iob = iob_all[self.patient_name]
        pid_df = pd.read_csv(self.pid_para_file)
        if patient_name not in pid_df.name.values:
            raise ValueError('{} not in PID csv'.format(patient_name))
        pid_params = pid_df.loc[pid_df.name == patient_name].squeeze()
        self.pid = PIDController(P=pid_params.kp, I=pid_params.ki, D=pid_params.kd, target=pid_params.setpoint)
        patient = T1DPatientNew.withName(patient_name, self.patient_para_file,
                                         random_init_bg=False, seed=self.seeds['patient'])
        sensor = CGMSensor.withName('Dexcom', self.sensor_para_file, seed=self.seeds['sensor'])
        if self.time_std is None:
            scenario = RandomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                              kind=self.kind, restricted=self.restricted_carb,
                                              harrison_benedict=self.harrison_benedict, unrealistic=self.unrealistic,
                                              deterministic_meal_size=self.deterministic_meal_size,
                                              deterministic_meal_time=self.deterministic_meal_time,
                                              deterministic_meal_occurrence=self.deterministic_meal_occurrence,
                                              meal_duration=self.meal_duration)
        elif self.use_custom_meal:
            scenario = CustomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                              num_meals=self.custom_meal_num, size_mult=self.custom_meal_size)
        else:
            scenario = SemiRandomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                                  time_std_multiplier=self.time_std, kind=self.kind,
                                                  harrison_benedict=self.harrison_benedict,
                                                  meal_duration=self.meal_duration)
        pump = InsulinPump.withName('Insulet', self.insulin_pump_para_file)
        self.env = _T1DSimEnv(patient=patient,
                              sensor=sensor,
                              pump=pump,
                              scenario=scenario,
                              sample_time=self.sample_time, source_dir=self.source_dir)
        if self.hist_init:
            self.env_init_dict = joblib.load("{}/{}_data.pkl".format(self.pid_env_path, self.patient_name))
            # The above PKL file is a dictionary with the following keys:
            # "state", "time", "time_hist", "bg_hist", "cgm_hist", "risk_hist", "lbgi_hist", "hbgi_hist",
            # "cho_hist", "insulin_hist".
            self.env_init_dict['magni_risk_hist'] = []
            for bg in self.env_init_dict['bg_hist']:
                self.env_init_dict['magni_risk_hist'].append(magni_risk_index([bg]))
            self._hist_init()

    def _hist_init(self):
        self.rolling = []
        env_init_dict = copy.deepcopy(self.env_init_dict)
        self.env.patient._state = env_init_dict['state']
        self.env.patient._t = env_init_dict['time']
        if self.start_date is not None:
            # need to reset date in start time
            orig_start_time = env_init_dict['time_hist'][0]
            new_start_time = datetime(year=self.start_date.year, month=self.start_date.month,
                                      day=self.start_date.day)
            new_time_hist = ((np.array(env_init_dict['time_hist']) - orig_start_time) + new_start_time).tolist()
            self.env.time_hist = new_time_hist
        else:
            self.env.time_hist = env_init_dict['time_hist']
        self.env.BG_hist = env_init_dict['bg_hist']
        self.env.CGM_hist = env_init_dict['cgm_hist']
        self.env.risk_hist = env_init_dict['risk_hist']
        self.env.LBGI_hist = env_init_dict['lbgi_hist']
        self.env.HBGI_hist = env_init_dict['hbgi_hist']
        self.env.CHO_hist = env_init_dict['cho_hist']
        self.env.insulin_hist = env_init_dict['insulin_hist']
        self.env.magni_risk_hist = env_init_dict['magni_risk_hist']

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a unit, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31
        return [seed1, seed2, seed3, seed4]

    @property
    def action_space(self):
        return spaces.Box(low=0, high=0.1, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
        if self.gt:
            return spaces.Box(low=0, high=np.inf, shape=(len(st),))
        else:
            num_channels = int(len(st)/self.state_hist)
            return spaces.Box(low=0, high=np.inf, shape=(num_channels, self.state_hist))










