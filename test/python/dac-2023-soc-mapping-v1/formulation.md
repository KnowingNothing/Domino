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