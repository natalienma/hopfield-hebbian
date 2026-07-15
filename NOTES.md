### backprop vs. hebbian learning
![alt text](image.png)

![alt text](neuron_synapse_anatomy.png)
![alt text](<Screenshot 2026-07-14 at 11.38.53 PM.png>)

### vocab:
neuron: a cell with 
1. an input wire/ dendrites (pre)
2. a body that adds everything (circle)
3. an output wire/axon (post)

spike: a brief electrical pulse the neuron sends down the axon once added up, if the sum crosses the threshold. if not, it doesn't get sent. binary.

synapse: the gap between the axon and dendrite of two neurons
- the thickness of the input/output line (axon/dendrite) = strength of activation

threshold: an input must surpass this to fire. thresholds are learned for every neuron. 
- adaptive threshold: a neuron's threshold can temporarily rise right ater it fires. also learnable

association: 
if neuron A fires, it will send the activation to B, which is added into B's total. Over time, if A and B keep firing together, the rule increases the weight at that connection. so it becomes more and more likely that A and B fire together.

### hyperparameters vs. trained parameters
hyperparameters:
- A+ and A- (strengthen/weaken amounts)
- tau (usually fixed)

trained parameters:
- synapse weights
- threshold (although usually fixed)

questions:
1. Over time, if B often fires after A, we increase the weight `w` between the two. Isn't this artificially inflating the number of times B will fire? Why not let A naturally influence B?
- `w` is randomly assigned at first. Updating it is just learning

2. why do we overwrite the entire weight matrix with new data every time? and why can we only overwrite 3%? if it was that easy, why haven't we been doing it before?

3. how can delta_t be negative when neuron B firing relies on neuron A to fire? 
- neuron B has many dendrites (inputs), so neuron A can be 0, and neuron B can still fire. 
- an SNN does not look like a typical ANN, where there are rounds, passes, or feed-forward. A and B are their own neurons, like people in a room that shout whenever they hear enough. There are no sequential layers like in ANNs.
- so if B fires before A, delta_t is negative. this just says that A does not cause B.

4. Do the edges/weights have direction?
- Yes. Just because A does not cause B, does not mean B does not cause A. B can cause A if B happens before A and the spikes are close in time.
- for this reason there is w_AB and w_BA, which can have two different values and STDP calculations

