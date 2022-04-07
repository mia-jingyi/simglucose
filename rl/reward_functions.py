def reward_risk(risk_hist, **kwargs):
    return -risk_hist[-1]


def reward_magni_risk(magni_risk_hist, **kwargs):
    return -magni_risk_hist[-1]
