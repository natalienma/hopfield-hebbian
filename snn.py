import numpy as np

dt = 0.001
tau_m = 0.02    # membrane time constant-- controls how fast a single neuron's voltage leaks over time
                # this is different from τ⁺/τ⁻
V_th = 1.0
V_reset = 0.0
beta = np.exp(-dt/tau_m) # produces a number between 0 and 1 (decay factor)

# neuron 1 simulation
total_time = 1.0
n_steps = int(total_time/dt)
v = 0.0
i = 0.3
spikes = []

for n in range(n_steps):
    v += i 
    if v >= V_th:
        spikes.append(1)
        v = V_reset
    else:
        spikes.append(0)