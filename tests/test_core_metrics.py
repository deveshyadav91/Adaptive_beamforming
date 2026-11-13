import numpy as np
from beamforming.array import ula_positions, steering_vector
from beamforming.adaptive import conventional_weights
from beamforming.metrics import (pattern_db,hpbw_deg_robust,visible_grating_lobes_ula,
                                  peak_sidelobe_level_db)

def test_half_wave_broadside_has_no_extra_grating_lobes():
    assert visible_grating_lobes_ula(0.5,0)==[]

def test_large_spacing_predicts_grating_lobe():
    assert len(visible_grating_lobes_ula(1.0,0))>=1

def test_pattern_metrics_are_finite():
    pos=ula_positions(12,0.5);theta=np.linspace(-90,90,1801)
    w=conventional_weights(steering_vector(pos,10))
    pat=pattern_db(pos,w,theta)
    bw=hpbw_deg_robust(theta,pat,10)
    sll=peak_sidelobe_level_db(theta,pat,10,8)
    assert 0 < bw < 90
    assert sll < 0
