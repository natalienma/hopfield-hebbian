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
    v = beta * v + i 
    if v >= V_th:
        spikes.append(1)
        v = V_reset
    else:
        spikes.append(0)

# Decay isn't a separate special event — it's baked into the update itself, 
# continuously. Reset is different: it only happens in the specific step a spike occurs, 
# and it's not "gradual" — it's an instant snap to 0. So the full picture: build up gradually 
# (leaking a bit each step) → cross threshold → instant snap to V_reset → 
# immediately start leaking/building up again from there. There's no gradual decay back down through the reset — 
# the reset itself is the discontinuity, decay only happens on the climb.

print(spikes)
print(sum(spikes))
