from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)
CONTROL_QUEST = 'simglucose/params/Quest.csv'
PATIENT_PARA_FILE = 'simglucose/params/vpatient_params.csv'
ParamTup = namedtuple('ParamTup', ['basal', 'cf', 'cr'])


class BBController(Controller):
    """
    This is a Basal-Bolus Controller that is typically practiced by a Type-1
    Diabetes patient. The performance of this controller can serve as a
    baseline when developing a more advanced controller.
    """
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min

        action = self._bb_policy(pname, meal, observation.CGM, sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        """
        Helper function to compute the basal and bolus amount.

        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance. 
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)
        
        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) + 
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 13.5, 23.52, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.info(f'Meal = {meal} g/min')
            logger.info(f'glucose = {glucose}')
            bolus = (
                (meal * env_sample_time) / quest.CR.values + (glucose > 150) *
                (glucose - self.target) / quest.CF.values).item()  # unit: U
        else:
            bolus = 0  # unit: U

        # This is to convert bolus in total amount (U) to insulin rate (U/min).
        # The simulation environment does not treat basal and bolus
        # differently. The unit of Action.basal and Action.bolus are the same
        # (U/min).
        bolus = bolus / env_sample_time  # unit: U/min
        return Action(basal=basal, bolus=bolus)

    def reset(self):
        pass


class ManualBBController(Controller):
    def __init__(self, target, cr, cf, basal, sample_rate=5, use_cf=True, use_bol=True, cooldown=0,
                 use_low_lim=False, low_lim=70):
        super().__init__(self)
        self.target = target
        self.orig_cr = self.cr = cr
        self.orig_cf = self.cf = cf
        self.orig_basal = self.basal = basal
        self.sample_rate = sample_rate
        self.use_cf = use_cf
        self.use_bol = use_bol
        self.cooldown = cooldown
        self.last_cf = np.inf
        self.use_low_lim = low_lim
        self.low_lim = low_lim

    def increment(self, cr_incr=0, cf_incr=0, basal_incr=0):
        self.cr += cr_incr
        self.cf += cf_incr
        self.basal += basal_incr

    def policy(self, observation, reward, done, **kwargs):
        carbs = kwargs.get('carbs')
        glucose = kwargs.get('glucose')
        action = self.manual_bb_policy(carbs, glucose)
        return action

    def manual_bb_policy(self, carbs, glucose, log=False):
        if carbs > 0:
            carb_correct = carbs / self.cr
            hyper_correct = (glucose > self.target) * (glucose - self.target) / self.cf
            hypo_correct = (glucose < self.low_lim) * (self.low_lim - glucose) / self.cf
            bolus = 0
            if self.use_low_lim:
                bolus -= hypo_correct
            if self.use_cf:
                if self.last_cf > self.cooldown and hyper_correct > 0:
                    bolus += hyper_correct
                    self.last_cf = 0
            if self.use_bol:
                bolus += carb_correct
            bolus = bolus / self.sample_rate
        else:
            bolus = 0
            carb_correct = 0
            hyper_correct = 0
            hypo_correct = 0
        self.last_cf += self.sample_rate
        if log:
            return Action(basal=self.basal, bolus=bolus), hyper_correct, hypo_correct, carb_correct
        else:
            return Action(basal=self.basal, bolus=bolus)

    def get_params(self):
        return ParamTup(basal=self.basal, cf=self.cf, cr=self.cr)

    def adjust(self, basal_adj, cr_adj):
        self.basal += self.orig_basal + basal_adj
        self.cr = self.orig_cr * cr_adj

    def reset(self):
        self.cr = self.orig_cr
        self.cf = self.orig_cf
        self.basal = self.orig_basal
        self.last_cf = np.inf