from .base import Controller
from .base import Action
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P, I, D, target=140, basal=None):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.basal = basal
        self.integral = 0
        self.prev_error = 0

    def step(self, value):
        error = self.target - value
        p_act = self.P * error
        self.integral += error
        logger.info('integrated error: {}'.format(self.integral))
        i_act = self.I * self.integral
        d_act = self.D * (error - self.prev_error)
        try:
            if self.basal is not None:
                b_act = self.basal
            else:
                b_act = 0
        except:
            b_act = 0
        self.prev_error = error
        logger.info('prev error: {}'.format(self.prev_error))
        control_input = p_act + i_act + d_act + b_act
        logger.info('Control input: {}'.format(control_input))

        # return the action
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        self.integral = 0
        self.prev_error = 0


def pid_test(pid, env, n_days, seed, full_save=False):
    env.seeds['sensor'] = seed
    env.seeds['scenario'] = seed
    env.seeds['patient'] = seed
    env.reset()
    full_patient_state = []
    for _ in tqdm(range(n_days*288)):
        act = pid.step(env.env.CGM_hist[-1])
        state, reward, done, info = env.step(action=act.basal)
        full_patient_state.append(info['patient_state'])
    full_patient_state = np.stack(full_patient_state)
    if full_save:
        return env.env.show_history(), full_patient_state
    else:
        return {'hist': env.env.show_history()[288:], 'kp': pid.P, 'ki': pid.I, 'kd': pid.D}