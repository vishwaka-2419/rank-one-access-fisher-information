#!/bin/bash
# Regenerate all numbers and figures added in the revision (AW12832).
# Requirements: Python >= 3.9, numpy, scipy, matplotlib.  Total run time: about 15 minutes on a laptop.
set -e
cd "$(dirname "$0")"
mkdir -p out figs
python3 R01_loss_crossover.py                   > out/R01_loss_crossover.txt
python3 R02_finite_window_small_chains.py       > out/R02_finite_window.txt
python3 R03_port_choice_deadline.py             > out/R03_port_choice.txt
python3 R04_disorder_perturbative.py            > out/R04_disorder_pt.txt
python3 R05_disorder_montecarlo.py              > out/R05_disorder_mc.txt
python3 R06_optimal_pulse.py                    > out/R06_optimal_pulse.txt
python3 R07_saturation_lindblad.py              > out/R07_saturation.txt
python3 R08_identity_nonreciprocity_thermal.py  > out/R08_identity.txt
python3 R09_microwave_example.py                > out/R09_microwave.txt
python3 R10_crosscheck_original.py              > out/R10_crosscheck.txt
python3 RF_loss_crossover.py
python3 RF_finite_window_data.py
python3 RF_finite_window.py
python3 RF_saturation.py
python3 RF_disorder.py
python3 RF_resource_rate.py
echo "done: text output in out/, figures in figs/"
