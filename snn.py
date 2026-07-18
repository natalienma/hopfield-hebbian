import numpy as np

dt = 0.001 # timestep = 1ms
tau_m = 0.02 # membrane time constant = R * C
V_th = 1 # threshold
V_reset = 0 # reset value
i = 1.5 # constant input current
total_time = 1.0
v = 0 # current voltage

beta = np.exp(-dt/tau_m)
spikes = []
n_steps = int(total_time/dt)
for n in range(n_steps):
    v = beta * v + i
    if v > V_th:
        spikes.append(1)
        v = V_reset
    else: 
        spikes.append(0)
    

