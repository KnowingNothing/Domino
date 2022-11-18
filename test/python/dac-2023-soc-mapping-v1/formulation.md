## Characterize the problem 
We have a computation graph $C(V_c,E_c)$ and accelerators $A$ also timing function $t: V_c \times A \to R$, resource functions $r_i: V_c \to R$, output volumn function $o: V_c \to R$, device constraint $M_i: V_a \to R$, communication cost function $c: A \times A \times R \to R$. 
We want to find a placement rule $p: V_c \to A$ and a scheduling rule $s: V_c \to R$ s.t.: 
- Resource constraint: tasks mapped onto streams of the device synchronously shared the resources;
  - $\forall t \in R, a \in A, i, \Sigma_{v\in V_c} 1_{t\in[S(v), S(v) + t(v, p(v))]} r_i(v) < M_i(a)$
- Dependency constraint: tasks dependency should not be voilated;
  - $\forall (v\to u) \in E_c, s(u,p(u)) \ge s(v,p(v))+t(v,p(v))+c(p(u), p(v), o(v))$ 
- The overall latency is optimized:
  - $p,s = argmin_{p,s}max_v \{s(v) + t(v, p(v))\}$

## DP formulation 

  - The dynamic programming algorithm needs two submodules, `grouper` and `placer`. The `grouper` finds several groups, subgraphs on the frontier of the computation graph, and pack them into independent jobs. `placer` gives a placement policy of the independent jobs onto accelerators under resource constraints. Upon execution, one global synchronization is performed after each group is executed. Suppose we have these components, the dynamic programming algoritm has update function is $DP(C) = max_{g\in grouper(C)}{profile(placer(g)) + DP(C - g)}$, the boundary condition is $DP(\emptyset) = 0$. 

  - grouper algorithm design 
    - Grouper needs to generate subgraph on the frontiers. To deal with this, we performs arbitrary topological sorts on the computation graph and selects the tail of the computation graph. 
  ```python 
  def group(G):
    for topo_orders in topological_sorts(G): 
      # a topo_order is a permutation of nodes based on topological sort algorithms like on DFS or BFS
      for num_node in range(num_node_min, num_node_max):
        # depth is the number node
        yield topo_order[:num_node]
  ```
  - placer algorithm design 
    - The placer deals with a traditional packing problem. Suppose we have $n$ resources, each job is a hyper-rectangle $(r^i_1,r^i_2,...,r^i_n,t_i) \forall i = 1,2,...,N$, the device is a pack $(Limit_1, Limit_2, ..., Limit_n, +inf)$, we need to put the hyper-rectengles into the pack and minimize the overall latency. This is a np-hard problem, we currently uses the a simple heuristic to solve this problem:
  ```python
  def placer(boxes: List[rectangles], pack: rectangle):
    boxes = sort(boxes, key = lambda box: box.time) # sort by times from low to high
    resource_usage = (0, 0, ..., 0)
    time_base = 0
    placement = Dict()
    for box in boxes:
      if resource_usage + box.resource_usage > resource_limit: 
        # +: (a,b) + (c,d) = (a+b, c+d); > (a, b) > (c,d) <=> a > c or b > d 
        time += last_time 
        resource_usage = 0
      else:
        resource_usage += box.resource_usage 
      placement[box] = (resource_usage, time)
    return time 
  ```

| Model | S1 | S2 | S3 | S4 | S5 | S6|
| - | - | - | - | - | - | - | 
|Resnet50|1 |  | |1 | | | 
|GoogLeNet| |  | | 1| | | 
|Unet| |  | | | | | 
|ModelNetV2|1 |  | | | | | 
|BR-Q Handpose| |  | | | | | 
|Focal Length DepthNet| |  | | | | | 
|SSD-Resnet34| |  | | 1| | | 
|SSD-MobileNetV1| |  | | | | | 
|GNMT| |  | |1 | | | 
|YoLoV3| |  | |1 | | | 
|EffitientNet-B0| |  | | | | | 
|MobileNet-v1| |  | | | | | 
|Tiny YOLO| |  | | | | | 
|Shufflenet|1 |  | | | | | 
|GPT2| | 1 | | | | | 
|MobileBert| | 1 | | | | | 
|Transformer XL| | 1 | | | | | 
|DLRM| |  |1 | | | | 
|WideDeep| | 1 | | | | | 
|NCF| |  | 1 | | | | 


# Experiments 
1. SoC 
  1. 2 conv (heterogeneous); 1 depthwise(fixed); 1 GEMM(fixed)
  2. 4 conv; 3 depthwise; 1 GEMM
2. workloads
  1. vision: resnet50; mobilenetV2; GoogleNet;
  2. NLP: ViT, Bert, Transformer(Encoder), GPT-2();
  3. Mixed: GoogleNet; yoloV5; Bert
3. Baseline:
  1. H2H: one stream; greedy mapping; 
  2. Megama: one stream GA; 
