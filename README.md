# DNC

![](./assets/dnc_nature_architecture.jpg)

This is my attempt at implementing Differentiable Neural Computer.

Differentiable Neural Computer (DNC) is the recent creation from Google DeepMind that was published in Nature under the name [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/nature/journal/v538/n7626/pdf/nature20101.pdf).

It's a recurrent neural network which includes a large number of preset operations that model various memory storage and management mechanisms.

In a way it is modular; DNC embeds another neural network inside it reffered to as the "controller".
Controller can anything that's differentiable: feedforward network, vanilla RNN, LSTM etc. 
In theory it is even possible to use DNC as a controller.
In practice we lack good tools for doing that (currently, TensorFlow is limited in this regard).

This implementation includes three tasks from the original paper: copy task, repeat copy task and bAbI question answering task.

TensorFlow 1.2.rc0 and Python 3.6 were used in the implementation.

---

### Copy and repeat copy tasks

Copy tasks are a sort of sanity check.
They're fast to run and easy to visualize.
The network is presented with a sequence of vectors and tasked to recall them entirely from memory, in the same order.
During the recall phase, no inputs are presented to the network in order to ensure that the network has actually stored all the vectors in memory (unlike in various char-rnn networks).

![](./assets/rcopy_input.png)
![](./assets/rcopy_output.png)

The sequences show above are sample input and output sequences from the repeat copy task. 
X axis represents time steps while Y axis represents elements of the vectors.

With repeat copy task it is possible to test DNC's dynamic memory allocation capabilities.

**Idea:** Make the number of memory slots lower than the total number of things DNC needs to remember.

Then the DNC needs to learn to reuse memory locations. By visualizing the write weightings and read weightings, it is possible to note several things:
* At each step, the writes are focused on a single location
* The focus changes with each step
* The focus on the write weightings corresponds to the focus on the read weightings
* The focus can never change to an already written location, unless that location has been read from after that write

![](./assets/write_weighting.png)
![](./assets/read_weighting.png)

It is further possible to analyze the internal state of DNC by plotting memory usage weightings. 
Note that the usage drops to zero after the network reads from that location.

![](./assets/usage_vectors.png)

Also note that in this specific example the network erroneously *doesn't* update the usage of the 6th location from top; resulting in network not using that memory location for the rest of the sequence. 


### bAbI synthetic question answering dataset

Error percentages of my DNC, baseline LSTM compared with DeepMind's results:

| Task | DNC | DeepMind's DNC | LSTM 256 | LSTM 512 | DeepMind LSTM 256 |
| -----|---------|------------------- |---|---|---|
| 1. 1 supporting fact |         | 9.0 &plusmn; 12.6                     | | | |
| 2. 2 supporting facts |         | 32.0 &plusmn; 20.5                   | | | |
| 3. 3 supporting facts |         | 39.6 &plusmn; 16.4                   | | | |
| 4. 2 argument rels. |         |  0.4 &plusmn; 0.7                  | | | |
| 5. 3 argument rels. |         | 1.5 &plusmn; 1.0                    | | | |
| 6. yes/no questions |         | 6.9 &plusmn; 7.5                    | | | |
| 7. counting |         | 9.8 &plusmn; 7.0                    | | | |
| 8. lists/sets |         | 5.5 &plusmn; 5.9                    | | | |
| 9. simple negation |         | 7.7 &plusmn; 8.3                    | | | |
| 10. indefinite knowl. |         | 9.6 &plusmn; 11.4                    | | | |
| 11. basic coreference |         | 3.3 &plusmn; 5.7                    | | | |
| 12. conjuction |         | 5.0 &plusmn; 6.3                    | | | |
| 13. compound coref. |         | 3.1 &plusmn; 3.6                    | | | |
| 14. time reasoning |         | 11.0 &plusmn; 7.5                    | | | |
| 15. basic deduction |         | 27.2 &plusmn; 20.1                    | | | |
| 16. basic induction |         | 53.6 &plusmn; 1.9                    | | | |
| 17. positional reas. |         | 32.4 &plusmn; 8.0                    | | | |
| 18. size reasoning |         | 4.2 &plusmn; 1.8                    | | | |
| 19. path finding |         | 64.6 &plusmn; 37.4                    | | | |
| 20. agent motiv. |         | 0.0 &plusmn; 0.1                    | | | |
| **Mean**        |      | 16.7 &plusmn; 7.6 |  | | |


bAbI dataset was notoriously slow to train with DNC, as the total training time was over 1 week on NVIDIA GTX 1080.
I was not able to raise the GPU utilization to more than 60%.


### Understanding memory operations

Interactions between the controller and memory and somewhat easy to understand.
However, operations that are being computer *in* the memory module are not.

Inspired by this [great DNC implementation](https://github.com/Mostafa-Samir/DNC-tensorflow) and the corresponding visualization of the DNC high level [data-flow diagrams](https://github.com/Mostafa-Samir/DNC-tensorflow/blob/master/docs/data-flow.md), I decided to create my own.

![](./assets/DNC_final.png)

The image represents *one* time step of DNC. Top arrow shows the general data flow. The dotted box represents the memory module.

There are many low-level, simple, differentiable memory operations in the memory module: cosine similarity, softmax, various compositions of multiplication and addition etc.

Those low level operations are composed in various ways which represent *something useful*. *Something useful* here means three attention mechanisms: content-based lookup, memory allocation and temporal memory linkage.

The attention mechanisms are parametrized by the three learnable weight matrices, whose corresponding MatMul operation is marked with the cross circle symbol.
The rest of the memory is fixed and, in a way, not subject to catastrophic forgetting.

### Things I don't understand 

##### DNC sometimes doesn't work
In certain scenarios, DNC performance seems to be much more dependent on weight initializations than performance of LSTM or similar recurrent architectures.
In other words, sometimes the loss doesn't converge and sometimes the loss doesn't seem like it's converging and then suddenly it drops to zero.

It seems to happen only on copy and repeat copy task when the network capacity is really low (low memory size/word size/controller capacity).

This implementation uses initialization from a normal distribution with 0.1 standard deviation.
Xavier and orthogonal initializations didn't seem to make a difference.

##### Sometimes there's NaN's

In the same scenarios the network tends to get NaN's in the computational graph. I've noticed it tends to happen after it gets on a completely wrong track and then is unable to solve it. 
Curriculum learning seems to help diminish it, but it still sometimes happens.

##### Gradient clipping seems to be needed

Without clipping, exploding gradients happen periodically, loss increases and network unlearns some of the things.
In bAbI task, the network immediatelly learns which type of words should the answer contain. 
Without clipping, sometimes it unlearned such a basic thing and it had the funny side effect of completely missing the point:

Input:

> wolves are afraid of cats . mice are afraid of wolves . cats are afraid of sheep . sheep are afraid of cats . gertrude is a mouse . jessica is a mouse . emily is a sheep. winona is a wolf . what is emily afraid of ? - what is gertrude afraid of ? - what is winona afraid of ? - what is jessica afraid of ? -

Output: 
> cat wolf football wolf

### A word on tensor contraction operations


This is a sales pitch for one operation this implementation relied heavily on: [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation).

It's a generalization of many various vector, matrix, and tensor operations. Dot product, matrix product, cross product, transpositions, summing across any axes; all those operations can be represented with the einsum operation.

I've found it incredibly useful not to have to reshape and transpose the tensors before using them.
There is a [great tutorial](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/) about it.

Although it is optimized in [numpy](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.einsum.html), TensorFlow still lacks a fast implementation.

[This one cool trick](https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.einsum.html) helps optimize it in TensorFlow.
I've gotten about 30-40% speed increase with it on bAbI tasks on NVIDIA GTX 1080.


