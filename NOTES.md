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

questions:
1. Over time, if B often fires after A, we increase the weight `w` between the two. Isn't this artificially inflating the number of times B will fire? Why not let A naturally influence B?
- `w` is randomly assigned at first. Updating it is just learning

2. why do we overwrite the entire weight matrix with new data every time? and why can we only overwrite 3%? if it was that easy, why haven't we been doing it before?
