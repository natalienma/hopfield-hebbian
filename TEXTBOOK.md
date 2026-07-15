# Spiking Neural Networks, Neuromorphic Hardware & Alternative Architectures
### Deep-dive interview prep notes

---

## 1. Spiking Neural Networks: the math

### 1.1 Why SNNs exist — the core reframe

An ANN neuron computes a dense, continuous-valued function every forward pass: `y = f(Wx + b)`. Every neuron fires *something* on every clock tick, dense or not.

A spiking neuron instead integrates an internal **state** (membrane potential) over time and only communicates when that state crosses a threshold — producing a **binary event** (a spike) rather than a real number. Communication is a train of delta functions:

$$s(t) = \sum_k \delta(t - t_k)$$

where $t_k$ are the neuron's firing times. This is the whole ballgame for energy: an ANN's synaptic array does a MAC for every weight on every input, every timestep, dense. An SNN only spends energy on a synapse when a spike actually crosses it. If activity is sparse in time, energy scales with the *number of spikes*, not the size of the weight matrix.

### 1.2 The Leaky Integrate-and-Fire (LIF) neuron

This is the workhorse model — know it cold.

**Continuous time:**

$$\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{rest}) + R\,I(t)$$

- $V(t)$: membrane potential
- $\tau_m = RC$: membrane time constant (leak rate)
- $I(t)$: input current, driven by incoming weighted spikes
- Fire-and-reset rule: if $V(t) \geq V_{th}$, emit a spike and reset ($V \to V_{reset}$, or $V \to V - V_{th}$ for a **soft reset**)

**Discretized (Euler) form** — this is what you actually implement/train:

$$V[t] = \beta\, V[t-1] + I[t] - S[t-1]\, V_{th}$$
$$S[t] = \Theta\big(V[t] - V_{th}\big)$$

where $\beta = e^{-\Delta t / \tau_m}$ is the decay factor (leak), $\Theta$ is the Heaviside step function, and the $-S[t-1]V_{th}$ term implements a **soft reset by subtraction** — it removes exactly the threshold's worth of charge rather than zeroing the potential. Soft reset is generally preferred in trained SNNs because it preserves "surplus" charge above threshold, which improves gradient flow and avoids information loss.

Input current from upstream spikes:

$$I[t] = \sum_j w_j\, S_j[t]$$

or, more realistically, filtered through a synaptic kernel (e.g. an alpha function or double-exponential) to model post-synaptic potential rise/decay rather than an instantaneous delta.

**Notice the structural resemblance to an RNN cell** — $V[t]$ depends recursively on $V[t-1]$, with a nonlinearity gating output. This is not a coincidence; it's central to how SNNs are trained (§1.5).

Other neuron models worth knowing by name, roughly in order of biological fidelity vs. tractability:
- **Izhikevich model**: 2D system reproducing many biological firing patterns (bursting, adaptation) with only 2 state variables — cheap and expressive.
- **Adaptive LIF (ALIF)**: adds a slow adapting threshold variable, gives the neuron working-memory-like dynamics, used heavily in recurrent SNNs.
- **Hodgkin-Huxley**: the "ground truth" ion-channel biophysics model, 4 coupled nonlinear ODEs — too expensive for any large-scale hardware or training, but it's the thing everything else is an approximation *of*. Good to mention you know it exists and why nobody uses it at scale.

### 1.3 Encoding: how do you get real-valued data into spikes?

This matters a lot for an energy-focused SNN startup because **the encoding scheme determines whether you actually get an energy win**.

- **Rate coding**: information in average firing rate over a window. Simple, robust, maps cleanly from ANN activations — but you need many timesteps per inference to estimate a rate accurately, which multiplies your operation count and can *erase* the sparsity advantage. This is the "dirty secret" of a lot of SNN benchmarks.
- **Temporal / latency coding (time-to-first-spike, TTFS)**: information in *when* the first spike occurs relative to stimulus onset. One spike per neuron can carry a lot of information — much better energy profile, but harder to train and more sensitive to noise.
- **Rank-order coding**: information in the *order* in which neurons fire, not exact timing.
- **Population coding**: a value is represented by the joint activity pattern across a population of neurons with overlapping tuning curves (this is very much a neuroscience import, e.g. how orientation is coded in visual cortex).
- **Delta / sigma-delta encoding**: encode *changes* in an analog signal as sparse spike events (this is exactly what event-based (DVS) cameras do at the sensor level, and what Loihi 2's graded-spike ANN-to-SNN conversion pipelines use — only sending spikes when a value changes by more than some threshold). This is probably the most directly relevant encoding scheme to ask about in an interview, since it's the one that produces real measured sparsity wins on hardware.

### 1.4 The training problem: spikes aren't differentiable

$\Theta(x)$ is a step function. Its derivative is a Dirac delta — zero almost everywhere, and undefined (infinite) exactly at threshold. If you try to backpropagate through the spike nonlinearity directly, gradients are zero everywhere except one immeasurable point. Standard backprop just doesn't work out of the box. This is *the* central technical challenge of the field, and any SNN company will expect you to understand why and what the workarounds are.

### 1.5 Surrogate gradient descent

The dominant modern solution: **use the real Heaviside function on the forward pass, but substitute a smooth surrogate function's derivative on the backward pass.**

Common surrogates (you should be able to write at least one from memory):

- **Fast sigmoid**: $\sigma(x) = \frac{x}{1+|x|}$, giving a smooth derivative used in place of $\Theta'(x)$
- **Arctangent surrogate**:
$$S(U) \approx \frac{1}{\pi}\arctan\!\left(\frac{\pi}{2}\alpha U\right) + \frac{1}{2}$$
with derivative
$$\frac{\partial S}{\partial U} = \frac{\alpha/2}{1 + \left(\frac{\pi}{2}\alpha U\right)^2}$$
used as the backward-pass gradient of the spike function ($\alpha$ controls sharpness of the transition — a tunable hyperparameter trading gradient magnitude vs. smoothness).
- **Straight-through estimator (STE)**: crudest option — just pretend the derivative is 1 in a window around threshold and 0 elsewhere (a box function).

Training then proceeds as **backpropagation through time (BPTT)**: unroll the network over all $T$ simulation timesteps, forward pass uses the real (non-differentiable) spike function, backward pass swaps in the surrogate derivative wherever $\partial S/\partial V$ appears in the chain rule. You also have to backprop through the leaky recurrence $V[t] = \beta V[t-1] + \dots$, which is structurally the same vanishing/exploding-gradient problem as vanilla RNNs (§3) — except now compounded with the spike-based information bottleneck.

An honest caveat worth raising in an interview: surrogate gradients work extremely well *empirically* (getting within 1–2% of ANN accuracy on many benchmarks) but the field still doesn't have a fully worked-out theoretical justification for *why* — it's understood as a heuristic bridge between smoothed-probabilistic-model gradients and true SNN dynamics, and is an active research topic. Knowing this nuance (rather than presenting surrogate gradients as a solved problem) will read as genuine depth.

### 1.6 Other training paradigms

- **ANN-to-SNN conversion**: train a normal ReLU network with backprop, then map weights onto an SNN where firing *rate* approximates the ReLU activation (rate coding, plus threshold/weight rebalancing so rates land in a usable range). Cheap and gets you SOTA-ish accuracy, but usually needs many timesteps per inference, which costs you both latency and the energy advantage you were trying to get in the first place. Good conversion pipelines use sigma-delta / graded spike coding specifically to claw back sparsity (Loihi 2's graded spikes were designed for exactly this).
- **STDP (Spike-Timing-Dependent Plasticity)**: local, unsupervised, biologically-derived Hebbian rule. Weight update depends only on the relative timing of pre- and post-synaptic spikes:
$$\Delta w = \begin{cases} A_+ \, e^{-\Delta t/\tau_+} & \Delta t > 0 \ \text{(pre before post — potentiate)} \\ -A_- \, e^{\Delta t/\tau_-} & \Delta t < 0 \ \text{(post before pre — depress)} \end{cases}$$
where $\Delta t = t_{post} - t_{pre}$. Fully local (no global error signal, no backward pass), which makes it extremely hardware-friendly (it's literally implementable as an analog circuit at the synapse), but it doesn't optimize a global task loss, so on its own it underperforms supervised methods on most benchmarks. Mostly used for unsupervised feature learning or on-chip continual/online learning.
- **Three-factor learning rules**: STDP (2-factor: pre- and post-synaptic activity) plus a third global modulatory signal (think dopamine/reward). This is what lets you do local, on-chip reinforcement learning — it's a headline feature of Loihi 2's learning engines, worth name-dropping.
- **Eligibility-trace / e-prop style methods**: approximate BPTT with a forward-only, online-computable trace, avoiding the need to store the whole unrolled computation graph. More biologically plausible and more hardware-implementable than full BPTT, at some accuracy cost. If the startup does *training on-chip* rather than off-chip-train/on-chip-deploy, this family is worth knowing.

### 1.7 Loss functions / readout

You still need to turn spike trains into a task loss:
- **Rate/count loss**: cross-entropy on spike counts (or average membrane potential) accumulated over the simulation window — treats the SNN like a "noisy ANN" and is the most common approach.
- **Membrane-potential readout**: skip the final spike nonlinearity on the output layer, do cross-entropy directly on the (non-spiking) membrane potential of output neurons — a common trick that sidesteps some of the reset/re-decode complexity.
- **TTFS / latency loss**: penalize the *time* of first output spike rather than a rate — directly optimizes for low-latency, low-spike-count inference, which is the metric you actually care about for energy, but is harder to optimize.

---

## 2. Neuromorphic hardware & the von Neumann question

### 2.1 The von Neumann bottleneck

Classic (von Neumann) architecture: one shared memory, one (or few) compute units, connected by a bus. Every operation is fetch-instruction → fetch-operand-from-memory → compute → write-back-to-memory. For a matrix multiply, this means every weight has to physically travel from DRAM/SRAM across a bus to the ALU for every use.

The key fact to internalize: **moving a bit of data across that bus costs vastly more energy than the arithmetic operation performed on it once it arrives** — often well over an order of magnitude, and the gap gets worse the farther off-chip the memory is (SRAM cache < on-chip DRAM < off-chip DRAM). This is *the* reason GPUs are so power-hungry at scale even though the actual multiply-accumulate is cheap — you're paying a data-movement tax on every single operation. This is the specific inefficiency neuromorphic and compute-in-memory architectures are designed to attack.

### 2.2 The non-von Neumann answer

Three design principles show up across essentially every neuromorphic chip:

1. **Co-location of memory and compute.** Synaptic weights live in local SRAM physically next to the accumulation logic on the same core ("neurocore"), so weight access doesn't cross a bus at all. Loihi 2's neurocores, for example, keep neuron-state SRAM and synaptic-weight SRAM directly local to the core's compute logic — this is what "compute-in-memory" means in the digital-neuromorphic sense (distinct from analog in-memory compute using memristor crossbars, which is a related but separate line of research).
2. **Event-driven, asynchronous execution.** No global clock forcing every circuit to toggle every cycle. A neurocore only does work when it receives a spike event; idle circuits draw near-zero dynamic power. Communication between cores happens via **Address-Event Representation (AER)** — spikes are transmitted as small digital packets encoding "which neuron fired," routed over an asynchronous mesh network-on-chip, rather than as dense synchronous bus traffic.
3. **Sparsity-proportional energy.** Because compute only happens on spikes, and because good encodings can make spikes rare, energy scales with the *number of spikes / synaptic operations actually generated*, not with the dense size of the weight matrix. This is the entire value proposition an SNN energy startup is selling — dense-matmul hardware (GPU/TPU) spends the same energy on a zero as on a large activation; event-driven hardware spends ~zero energy on a silent neuron.

A useful framing for the interview: **von Neumann vs. non-von Neumann is really "data-movement-dominated vs. sparsity-and-locality-dominated" energy accounting.** GPUs have gotten good at amortizing data movement (large batches, high arithmetic intensity, big on-chip caches) but they're still fundamentally paying to move dense tensors around a shared-memory hierarchy. Neuromorphic chips trade that for an architecture that's *only* efficient if your workload is genuinely sparse and event-like — which is exactly why the honest pitch for these chips is about specific workload classes (always-on sensing, robotics perception, edge audio/vision), not "replace GPUs for training transformers."

### 2.3 The chip landscape (know a few concretely)

| Chip | Type | Notes |
|---|---|---|
| **Intel Loihi (2018)** | Digital, spiking | 14nm, 128 cores, ~130k neurons / 130M synapses, on-chip STDP, ~30 billion synaptic ops/sec at roughly 15 pJ per synaptic operation, embedded x86 cores for housekeeping tasks |
| **Intel Loihi 2** | Digital, spiking | Intel 4 (7nm)-class process, up to ~128 cores, ~1M neurons / up to 120M synapses per chip, **graded (integer-valued) spikes** not just binary — lets it emulate sigma-delta / ANN-conversion coding for extra sparsity, native **three-factor learning rules**, ~1W typical chip power, reported >100× energy efficiency vs. CPU and ~30× vs. GPU on SNN workloads, sub-millijoule inference on sensor-fusion tasks |
| **IBM TrueNorth (2014)** | Digital, spiking | 4096 cores, 1M neurons / 256M synapses, ~70mW, fully digital crossbar-per-core design, weights fixed after offline training (no on-chip learning) — historically important, largely superseded now |
| **IBM NorthPole** | Digital, **non-spiking** compute-in-memory | Not an SNN chip — a frame-based inference accelerator that eliminates DRAM entirely by keeping the whole model in on-chip SRAM. Worth knowing because it shows the "avoid von Neumann bottleneck" idea generalizes beyond spiking representations specifically — co-location of memory and compute is the deeper principle, spiking is one strategy for exploiting it, not the only one |
| **SpiNNaker / SpiNNaker2** | Digital, spiking, many-core (ARM-based) | Built for massive-scale, real-time brain simulation rather than max energy efficiency per se; less "compute-in-memory," more "many small general cores + efficient spike routing fabric" |
| **BrainScaleS** | Analog/mixed-signal, spiking | Physical (not digital-simulated) neuron circuits, runs *faster than biological real-time* — a genuinely different design philosophy (analog physical modeling vs. digital simulation) |
| **BrainChip Akida** | Digital, spiking, commercial | Shipping at volume for edge vision/audio, event-based CNN-ish workloads, on-chip learning, milliwatt-to-sub-milliwatt power — the clearest example of a neuromorphic chip that's a real commercial product today rather than a research platform |

A distinction worth having ready: **not everything called "neuromorphic" is spiking, and not everything spiking is analog.** The useful axes are (a) spiking vs. frame/event-based representation, and (b) digital vs. analog/mixed-signal implementation — Loihi and TrueNorth are digital+spiking, BrainScaleS is analog+spiking, NorthPole is digital+non-spiking-but-still-non-von-Neumann. If a startup says "neuromorphic," ask which of these axes they actually mean.

---

## 3. Recurrent networks — the math, and why they matter here

You know transformers; RNNs matter for this interview because **SNNs are, mathematically, a special (nonlinear, discrete, sparse-output) case of recurrent computation**, and a lot of SNN training theory is directly inherited from RNN training theory.

### 3.1 Vanilla RNN

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Trained with **BPTT**: unroll over $T$ steps, backprop the loss through the recurrence. The gradient of the loss at step $T$ with respect to $h_t$ involves a product of $T-t$ Jacobians:

$$\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T}\prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Each Jacobian factor is roughly $\text{diag}(\tanh'(\cdot))\, W_{hh}$. If the largest eigenvalue of $W_{hh}$ (scaled by the local derivative) is consistently $<1$, the product shrinks geometrically → **vanishing gradients**, long-range dependencies can't be learned. If it's consistently $>1$ → **exploding gradients**, training diverges (usually handled with gradient clipping). This is *exactly* the same mechanism you saw in §1.2/1.5 for the LIF neuron's leak term $\beta$ compounding over $T$ timesteps — an SNN trained with BPTT inherits this problem, and it's part of why very deep (in time) SNNs are hard to train.

### 3.2 LSTM

Fixes vanishing gradients with an additive, gated memory cell:

$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update — additive!)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

The key structural trick: $c_t$ updates *additively* ($c_t = f_t \odot c_{t-1} + \dots$) rather than through a repeated matrix multiply + squashing nonlinearity, so the gradient path $\partial c_t/\partial c_{t-1} = f_t$ doesn't force geometric decay the way $\tanh'(\cdot)W_{hh}$ does — as long as the forget gate stays near 1, gradients can flow across long time spans largely unimpeded.

### 3.3 GRU

A leaner alternative merging the forget/input gates and dropping the separate cell state:

$$z_t = \sigma(W_z[h_{t-1},x_t]), \quad r_t = \sigma(W_r[h_{t-1},x_t])$$
$$\tilde h_t = \tanh(W[r_t \odot h_{t-1}, x_t]), \quad h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t$$

Similar idea — an interpolation ("update gate" $z_t$) between old and new state rather than a hard overwrite.

### 3.4 The RNN ↔ SNN connection, explicitly

Compare the LIF update again: $V[t] = \beta V[t-1] + I[t] - S[t-1]V_{th}$. This is structurally an RNN cell where:
- the "hidden state" is the membrane potential,
- the recurrence weight is a fixed scalar leak $\beta$ (not learned, in the simplest LIF — though **adaptive** variants do learn per-neuron time constants),
- the "nonlinearity" is a hard threshold rather than tanh/sigmoid,
- the **output at every timestep is binary and sparse**, rather than a dense real-valued vector.

That last point is the whole story: an SNN is an RNN whose hidden-to-output map has been replaced with something that (a) is non-differentiable, forcing surrogate gradients, and (b) produces sparse binary events instead of dense reals, which is what makes it a plausible target for event-driven hardware. If you can talk fluently about vanishing gradients, gating, and BPTT for RNNs, you can map every one of those ideas one-to-one onto SNN training — that mapping is a strong thing to demonstrate in an interview.

---

## 4. Alternative / efficiency-oriented architectures (context for "why not just use X")

Good to know for the inevitable "how does this compare to state space models / linear attention" question — SNNs are one point in a broader landscape of architectures trying to beat quadratic-attention-and-dense-matmul costs.

- **State Space Models (S4, Mamba)**: model sequences via a linear time-invariant (or, in Mamba's case, input-*dependent*/selective) state-space recursion $h_t = \bar A h_{t-1} + \bar B x_t,\; y_t = C h_t$, computed efficiently either as a global convolution (S4, via a structured/HiPPO-initialized $A$) or as a parallel scan (Mamba's selective SSM). Linear time and memory in sequence length, versus attention's quadratic cost. Conceptually close to an RNN, and *structurally close to a non-spiking LIF neuron* — both are linear leaky recurrences — but SSMs keep continuous-valued states/outputs and are trained with dense backprop, whereas SNNs binarize the output. Worth explicitly contrasting: SSMs chase *sequence-length efficiency on dense hardware*, SNNs chase *event-sparsity efficiency on event-driven hardware* — different axis of "efficient."
- **RWKV**: an RNN-formulated model designed to be trainable in a parallelizable, attention-like form and run as an efficient recurrence at inference — another "get RNN-style linear inference cost without giving up parallel training" approach.
- **Liquid Neural Networks / Liquid Time-Constant networks**: continuous-time RNNs where the time constant itself is input-dependent (an ODE-based neuron, closer in spirit to biological neurons and to LIF dynamics than a standard RNN cell) — small, adaptive, good for physical/control/edge time-series settings; conceptually a cousin of SNNs in that both take neuronal *time dynamics* seriously rather than treating time as just "another sequence index."
- **Linear attention / Performer / retentive networks**: reformulate the attention kernel to avoid materializing the full $O(n^2)$ attention matrix, trading exactness for linear-in-sequence-length cost — same motivating problem as SSMs (quadratic attention cost), different mechanism.

The throughline to be able to articulate: **transformers spend fixed dense compute regardless of content; SSMs/RNNs spend compute linear in sequence length regardless of content; SNNs spend compute proportional to how much the content actually changes/fires, and only realize that advantage on hardware built to skip idle neurons.** That's the specific bet an SNN-for-energy startup is making, and it's worth being ready to discuss both why it's compelling (genuine sub-linear, content-dependent compute) and where it's fragile (training difficulty, rate-coding overhead eating the sparsity gain, immature toolchains vs. the CUDA ecosystem).

---

## 5. Likely interview threads to be ready for

- Derive the LIF discrete-time update and explain reset-by-subtraction vs. reset-to-zero.
- Explain *why* backprop fails on spikes and walk through a surrogate gradient by name and formula.
- Compare rate coding vs. temporal coding energy/accuracy tradeoffs — and be ready to say rate coding can *lose* the energy advantage if timestep count is high.
- Explain von Neumann bottleneck in terms of data-movement energy vs. compute energy, not just "memory and compute are separate."
- Name at least 2–3 real chips and one concrete number for each (cores, neurons/synapses, pJ/op or W).
- Connect SNN training difficulty explicitly to RNN vanishing-gradient / BPTT concepts.
- Be ready to honestly discuss where the "SNNs save energy" claim is fragile (encoding overhead, immature compilers/toolchains, accuracy gap vs. ANNs on some tasks, conversion vs. direct-training tradeoffs) — companies in this space generally want people who understand the real bottlenecks, not just the pitch.