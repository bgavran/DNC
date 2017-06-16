# DNC

This is my attempt at implementing Differentiable Neural Computer.

Differentiable Neural Computer (DNC) is the recent creation from Google DeepMind that was published in Nature under the name [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/nature/journal/v538/n7626/pdf/nature20101.pdf).

It's a recurrent neural network which includes a large number of preset operations that model various memory storage and management mechanisms.

In a way it is modular; DNC embeds another, controller neural network inside it.
Controller can anything that's differentiable: feedforward network, vanilla RNN, LSTM etc. 
In theory it is even possible to use DNC as a controller.
In practice we lack good tools for doing that (currently, TensorFlow is limited in this regard).

This implementation includes three tasks from the original paper: copy task, repeat copy task and bAbI question answering task.

TensorFlow 1.2.rc0 and Python 3.6 were used in the implementation.

---

### Copy and repeat copy tasks

![](./assets/rcopy_input.png)
![](./assets/rcopy_output.png)
![](./assets/write_weighting.png)
![](./assets/read_weighting.png)
![](./assets/usage_vectors.png)



### bAbI synthetic question answering dataset

| Task | Error % | DeepMind's error % |
| -----|---------|------------------- |
| 1. 1 supporting fact |         | 9.0 &plusmn; 12.6                     |
| 2. 2 supporting facts |         | ...                   |
| 3. 3 supporting facts |         |                    |
| 4. 2 argument rels. |         |                    |
| 5. 3 argument rels. |         |                    |
| 6. yes/no questions |         |                    |
| 7. counting |         |                    |
| 8. lists/sets |         |                    |
| 9. simple negation |         |                    |
| 10. indefinite knowl. |         |                    |
| 11. basic coreference |         |                    |
| 12. conjuction |         |                    |
| 13. compound coref. |         |                    |
| 14. time reasoning |         |                    |
| 15. basic deduction |         |                    |
| 16. basic induction |         |                    |
| 17. positional reas. |         |                    |
| 18. size reasoning |         |                    |
| 19. path finding |         |                    |
| 20. agent motiv. |         |                    |


### Understanding memory operations

Inspired by this [great DNC implementation](https://github.com/Mostafa-Samir/DNC-tensorflow) and the corresponding visualization of the DNC high level [data-flow diagrams](https://github.com/Mostafa-Samir/DNC-tensorflow/blob/master/docs/data-flow.md), I decided to create my own.

Interactions between the controller and memory and somewhat easy to understand.
However, operations that are being computer *in* the memory module are much more difficult to understand.

![](./assets/DNC_final.png)

The image represents *one* time step of DNC. Top arrow shows the general data flow.

There are many low-level, simple, differentiable memory operations in the memory module: cosine similarity, softmax, various compositions of multiplication and addition etc.

Those low level operations are composed in various ways which represent *something useful*. *Something useful* here means three attention mechanisms: content-based lookup, memory allocation and temporal memory linkage.

The attention mechanisms are parametrized by the learnable weight matrices, whose corresponding MatMul operation is marked with the cross circle symbol.
The rest of the memory is fixed and, in a way, not subject to catastrophic forgetting.



