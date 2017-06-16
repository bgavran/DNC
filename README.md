# DNC

This is my attempt at implementing Differentiable Neural Computer.

Differentiable Neural Computer (DNC) is the recent creation from Google DeepMind that was published in Nature under the name [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/nature/journal/v538/n7626/pdf/nature20101.pdf).

It's a recurrent neural network which includes a large number of preset operations that model various memory storage and management mechanisms.

In a way it is modular; DNC embeds another, controller neural network inside it.
Controller can anything that's differentiable: feedforward network, vanilla RNN, LSTM etc. 
In theory it is possible to use DNC as a controller and, in a way, recursively nest them. 
In practice we lack good tools for doing that (currently, TensorFlow is limited in this regard).

This implementation includes three tasks from the original paper: copy task, repeat copy task and bAbI question answering task.

TensorFlow 1.2.rc0 and Python 3.6 were used in the implementation.
---

### Copy and repeat copy tasks

![](./img/rcopy_intput.png)
![](./img/rcopy_output.png)
![](./img/write_weighting.png)
![](./img/read_weighting.png)
![](./img/usage_vectors.png)



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








