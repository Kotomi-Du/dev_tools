# Summary
This PR is to introduce output-rooted DFS for in-order queue. It can improve memory-pool reuse and lowers peak memory by places single-consumer Transpose, Convert, and Reorder nodes close to consumers. An example of the improvement is LLM pipeline which has weights(~4GB) as parameters can reduce memory from 15GB to ~11GB.

The solution only applies for in-order queue to minimize the impact of out-of-order queues which requires more parallel work.

# Problem
Current GPU processing order is based on input-rooted DFS order of the primitive dependency graph.

graph_initializations calls calc_processing_order() in graph_initializations.cpp:62.
The initial order is produced by DFS from input nodes. Users are visited recursively, then nodes are inserted at the front, ensuring dependencies precede users: nodes_ordering.cpp:13-34.
It is then reordered by graph depth using calculate_BFS_processing_order(): nodes_ordering.cpp:42-72. Each node gets its maximum distance from an input, and nodes are grouped from shallowest to deepest. Within one depth, the previous topological order is preserved.

Optimization passes may insert, remove, or fuse nodes while maintaining this list.
After all transformations, BFS ordering is recalculated for an out-of-order queue
With such design, for the topology below, A shallow/input-oriented processing order becomes:
```
T1 → T2 → T3 → A1 → M1 → A2 → M2 → A3 → M3
```


This creates the following output-buffer lifetimes:
```
Processing step:  1   2   3   4   5   6   7   8   9
                  T1  T2  T3  A1  M1  A2  M2  A3  M3

T1 output:        [===============]
T2 output:            [===================]
T3 output:                [=======================]
```
This leads to a problem that memory cannot be reused effectively. The peak memory is Mpeak = M1+M2+M3.

Solution
Introduce output-rooted DFS with the long dependency visited first produces:
A1 → T1 → M1 → A2 → T2 → M2 → A3 → T3 → M3
The lifetimes then become nearly disjoint:
```
T1 output:    [T1 → M1]
T2 output:                 [T2 → M2]
T3 output:                              [T3 → M3]
```
The memory pool can reuse one allocation: Mpeak =max(M1,M2,M3)