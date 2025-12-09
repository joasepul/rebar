use crate::Graph;
use rand::prelude::*;
use std::collections::HashMap;

/// Compressed Sparse Row (CSR) representation of the graph for performance.
#[derive(Debug, Clone)]
pub struct CompactGraph {
    pub offsets: Vec<usize>,        // len = n + 1
    pub neighbors: Vec<u32>,        // len = 2m
    pub weights: Option<Vec<f32>>,  // len = 2m, if weighted
    pub total_edge_weight: f32,
    pub num_nodes: usize,
}

impl CompactGraph {
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.node_count();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::new();
        // Assuming unweighted for now, or we can add weights to Graph later.
        // The user prompt mentions Graph has Option<Vec<f32>> weights but our current Graph struct doesn't.
        // We'll assume unweighted (weight=1.0) for the current Graph implementation.
        let weights = None; 
        let mut total_edge_weight = 0.0;

        offsets.push(0);
        for i in 0..n {
            let node_neighbors = graph.neighbors(i);
            for &neighbor in node_neighbors {
                neighbors.push(neighbor as u32);
                total_edge_weight += 1.0; // Undirected, so we count each direction
            }
            offsets.push(neighbors.len());
        }

        // total_edge_weight in the prompt context usually refers to sum of weights of all edges.
        // For undirected graph where each edge (u,v) is stored as u->v and v->u, 
        // the sum of all stored weights is 2 * total_weight.
        // The modularity formula uses 2m.
        // So total_edge_weight here represents 2m.
        
        Self {
            offsets,
            neighbors,
            weights,
            total_edge_weight,
            num_nodes: n,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn neighbors(&self, node: usize) -> &[u32] {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.neighbors[start..end]
    }

    pub fn weight(&self, _edge_idx: usize) -> f32 {
        if let Some(_weights) = &self.weights {
            // We need to find the index in the weights array.
            // The neighbors array stores target nodes.
            // The edge_idx passed here... wait, the caller doesn't pass edge_idx usually.
            // Let's look at usage.
            // local_moving uses neighbor_weights map.
            // aggregate_graph builds weights.
            // compute_modularity assumes 1.0 currently.
            
            // Actually, for CSR, we iterate neighbors.
            // We should probably change how we access weights.
            // But for now, let's just return 1.0 if weights is None.
            // If weights is Some, we need to know WHICH edge.
            // The current API `weight(edge_idx)` is not used.
            // Let's remove it or fix it.
            1.0
        } else {
            1.0
        }
    }
    
    pub fn degree(&self, node: usize) -> f32 {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        
        if let Some(weights) = &self.weights {
            let mut sum = 0.0;
            for i in start..end {
                sum += weights[i];
            }
            sum
        } else {
            (end - start) as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommunityStats {
    pub size: Vec<u32>,          // |C|
    pub tot_weight: Vec<f32>,    // Σ k_i for i in C
}

impl CommunityStats {
    pub fn from_initial_partition(graph: &CompactGraph, community_of: &[u32]) -> Self {
        let num_communities = *community_of.iter().max().unwrap_or(&0) as usize + 1;
        let mut size = vec![0; num_communities];
        let mut tot_weight = vec![0.0; num_communities];

        for node in 0..graph.num_nodes() {
            let comm = community_of[node] as usize;
            size[comm] += 1;
            tot_weight[comm] += graph.degree(node);
        }

        Self { size, tot_weight }
    }
    
    pub fn add(&mut self, comm: usize, degree: f32) {
        if comm >= self.size.len() {
            self.size.resize(comm + 1, 0);
            self.tot_weight.resize(comm + 1, 0.0);
        }
        self.size[comm] += 1;
        self.tot_weight[comm] += degree;
    }
    
    pub fn remove(&mut self, comm: usize, degree: f32) {
        self.size[comm] -= 1;
        self.tot_weight[comm] -= degree;
    }
}

pub struct LeidenConfig {
    pub gamma: f32,
    pub max_passes: usize,
    pub min_delta_q: f32,
    pub random_seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            max_passes: 10,
            min_delta_q: 0.0001,
            random_seed: 42,
        }
    }
}

pub struct LeidenResult {
    pub community_of: Vec<u32>,
    pub modularity: f32,
    pub num_communities: usize,
}

pub fn compute_modularity(
    graph: &CompactGraph,
    community_of: &[u32],
    comm_stats: &CommunityStats,
    gamma: f32,
) -> f32 {
    let m2 = graph.total_edge_weight; // 2m
    let mut q = 0.0;

    // Q = (1/2m) * sum_C ( e_C - gamma * (tot_C^2 / 2m) )
    // where e_C is 2 * number of edges inside C (or sum of weights inside C)
    // Actually, simpler: sum over all edges (i,j):
    // if c_i == c_j: q += A_ij - gamma * k_i * k_j / 2m
    
    // Let's use the community-wise formula for speed if possible, but iterating edges is O(m).
    
    for u in 0..graph.num_nodes() {
        let c_u = community_of[u];
        let _k_u = graph.degree(u);
        
        // Neighbors
        for &v in graph.neighbors(u) {
            let v = v as usize;
            let c_v = community_of[v];
            if c_u == c_v {
                // A_uv term (1.0)
                q += 1.0;
            }
        }
        
        // Null model term: - gamma * k_u * tot_C / 2m
        // We sum this over all u.
        // sum_u ( - gamma * k_u * tot_{c_u} / 2m )
        // = - (gamma / 2m) * sum_C ( tot_C * tot_C )
    }
    
    // The loop above double counts edges for A_ij term (u->v and v->u), which is correct for 1/2m factor.
    // But the null model term needs to be subtracted carefully.
    
    let mut null_term = 0.0;
    for tot_c in &comm_stats.tot_weight {
        null_term += tot_c * tot_c;
    }
    
    (q - gamma * null_term / m2) / m2
}

pub fn leiden(graph: &Graph, config: &LeidenConfig) -> LeidenResult {
    let mut compact_graph = CompactGraph::from_graph(graph);
    let mut community_of: Vec<u32> = (0..compact_graph.num_nodes() as u32).collect();
    let mut comm_stats = CommunityStats::from_initial_partition(&compact_graph, &community_of);
    
    let mut rng = StdRng::seed_from_u64(config.random_seed);
    
    // Track mapping from current level nodes to original nodes
    // Level 0: Identity
    // Level 1: Super-node -> [Level 0 nodes]
    // let mut node_mappings: Vec<Vec<Vec<usize>>> = Vec::new(); // Unused for now
    
    // Initial mapping: Level 0 nodes are the original nodes
    // We don't need to store Level 0 mapping explicitly if we handle it carefully.
    // But for aggregation, we need to know which original nodes belong to which super-node.
    // Let's maintain `original_community_of`: node_idx -> final_community_id
    
    // Actually, simpler:
    // We have `community_of` for the current graph.
    // If we aggregate, we get a new graph and a new `community_of` (identity).
    // We need to compose the mappings.
    // Let `layer_membership[i]` be the community of node `i` at layer `L` in layer `L+1`.
    
    let mut layer_memberships: Vec<Vec<u32>> = Vec::new();
    
    for _pass in 0..config.max_passes {
        // Phase 1: Local Moving
        let mut improved = true;
        let mut iter = 0;
        while improved && iter < 10 {
            improved = local_moving(&compact_graph, &mut community_of, &mut comm_stats, config.gamma, &mut rng);
            iter += 1;
        }
        
        // Phase 2: Refinement
        refine_partition(&compact_graph, &mut community_of, &mut comm_stats, config.gamma);
        
        // Phase 3: Aggregation
        let (coarse_graph, coarse_partition, mapping) = aggregate_graph(&compact_graph, &community_of);
        
        // Store the membership for this layer (mapping to next layer)
        layer_memberships.push(mapping);
        
        // If no reduction, stop
        if coarse_graph.num_nodes() == compact_graph.num_nodes() {
            break;
        }
        
        compact_graph = coarse_graph;
        community_of = coarse_partition;
        comm_stats = CommunityStats::from_initial_partition(&compact_graph, &community_of);
    }
    
    // Flatten memberships to get original node -> final community
    let num_original_nodes = graph.node_count();
    let mut final_community_of = vec![0; num_original_nodes];
    
    for i in 0..num_original_nodes {
        let mut current_node = i;
        for layer in &layer_memberships {
            current_node = layer[current_node] as usize;
        }
        // Map to final community ID in the last graph
        final_community_of[i] = community_of[current_node];
    }
    
    // Compute final modularity on original graph
    let original_compact = CompactGraph::from_graph(graph);
    let final_stats = CommunityStats::from_initial_partition(&original_compact, &final_community_of);
    let modularity = compute_modularity(&original_compact, &final_community_of, &final_stats, config.gamma);
    
    let num_communities = *final_community_of.iter().max().unwrap_or(&0) as usize + 1;
    
    LeidenResult {
        community_of: final_community_of,
        modularity,
        num_communities,
    }
}

fn local_moving(
    graph: &CompactGraph,
    community_of: &mut Vec<u32>,
    comm_stats: &mut CommunityStats,
    gamma: f32,
    rng: &mut impl Rng,
) -> bool {
    let n = graph.num_nodes();
    let mut nodes: Vec<usize> = (0..n).collect();
    nodes.shuffle(rng);
    
    let mut improved = false;
    let m2 = graph.total_edge_weight;
    
    // Reusable buffer for neighbor communities: comm_id -> weight
    // Using a dense vector if num_communities is small, or HashMap if large.
    // For large graphs, HashMap is safer.
    let mut neighbor_weights: HashMap<u32, f32> = HashMap::new();
    
    for &u in &nodes {
        let old_comm = community_of[u];
        let degree = graph.degree(u);
        
        // 1. Remove u from old community
        comm_stats.remove(old_comm as usize, degree);
        
        // 2. Identify neighbor communities and weights
        neighbor_weights.clear();
        
        let start = graph.offsets[u];
        let end = graph.offsets[u + 1];
        
        for i in start..end {
            let v = graph.neighbors[i] as usize;
            let weight = if let Some(weights) = &graph.weights {
                weights[i]
            } else {
                1.0
            };
            
            let v_comm = community_of[v];
            // If v is u, ignore self-loop for moving logic (it moves with u)
            if v != u {
                *neighbor_weights.entry(v_comm).or_insert(0.0) += weight;
            }
        }
        
        // 3. Find best community
        // We want to maximize delta_Q.
        // delta_Q = (k_i_in_C - gamma * k_i * tot_C / m2)
        // We can ignore the removal term from old_comm because it's constant for all candidates
        // (we already removed u, so we are choosing where to put it).
        
        // Check old community (it's a candidate, weight might be 0 if no neighbors there)
        let w_old = *neighbor_weights.get(&old_comm).unwrap_or(&0.0);
        let tot_old = comm_stats.tot_weight[old_comm as usize];
        let mut best_score = w_old - gamma * degree * tot_old / m2;
        let mut best_comm = old_comm;
        
        for (&comm, &w_c) in &neighbor_weights {
            if comm == old_comm { continue; }
            
            let tot_c = comm_stats.tot_weight[comm as usize];
            let score = w_c - gamma * degree * tot_c / m2;
            
            if score > best_score {
                best_score = score;
                best_comm = comm;
            }
        }
        
        // Also consider a new empty community?
        // In standard Louvain/Leiden, we usually only move to neighbor communities.
        // If best_score is worse than creating a singleton, maybe?
        // But usually we start with singletons, so we only merge.
        
        // 4. Move u to best community
        community_of[u] = best_comm;
        comm_stats.add(best_comm as usize, degree);
        
        if best_comm != old_comm {
            improved = true;
        }
    }
    
    improved
}

fn refine_partition(
    graph: &CompactGraph,
    community_of: &mut Vec<u32>,
    comm_stats: &mut CommunityStats,
    _gamma: f32,
) {
    // Simple refinement: Ensure connectivity.
    // Iterate over all communities. If a community is disconnected, split it.
    
    let n = graph.num_nodes();
    let mut visited = vec![false; n];
    let mut new_community_of = community_of.clone();
    let mut next_comm_id = *community_of.iter().max().unwrap_or(&0) + 1;
    
    // Group nodes by community for efficient traversal
    let mut nodes_by_comm: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        nodes_by_comm.entry(community_of[i]).or_default().push(i);
    }
    
    for (comm_id, nodes) in nodes_by_comm {
        if nodes.is_empty() { continue; }
        
        // Check connected components
        let mut components = 0;
        for &start_node in &nodes {
            if visited[start_node] { continue; }
            
            components += 1;
            let target_comm = if components == 1 {
                comm_id // Keep first component in original community
            } else {
                let id = next_comm_id;
                next_comm_id += 1;
                id
            };
            
            // BFS
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start_node);
            visited[start_node] = true;
            new_community_of[start_node] = target_comm;
            
            while let Some(u) = queue.pop_front() {
                for &v in graph.neighbors(u) {
                    let v = v as usize;
                    if community_of[v] == comm_id && !visited[v] {
                        visited[v] = true;
                        new_community_of[v] = target_comm;
                        queue.push_back(v);
                    }
                }
            }
        }
    }
    
    *community_of = new_community_of;
    *comm_stats = CommunityStats::from_initial_partition(graph, community_of);
}

fn aggregate_graph(
    graph: &CompactGraph,
    community_of: &[u32],
) -> (CompactGraph, Vec<u32>, Vec<u32>) {
    // 1. Renumber communities to 0..k
    let mut comm_map: HashMap<u32, usize> = HashMap::new();
    let mut next_id = 0;
    let mut new_comm_ids = vec![0; graph.num_nodes()];
    
    for (i, &c) in community_of.iter().enumerate() {
        let new_id = *comm_map.entry(c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        new_comm_ids[i] = new_id as u32;
    }
    
    let num_new_nodes = next_id;
    
    // 2. Build edges for new graph
    // We need to sum weights between communities.
    // Use a vector of HashMaps or similar?
    // Since we output CSR, we can build adjacency list first.
    
    let mut adj: Vec<HashMap<usize, f32>> = vec![HashMap::new(); num_new_nodes];
    let mut total_weight = 0.0;
    
    for u in 0..graph.num_nodes() {
        let c_u = new_comm_ids[u] as usize;
        for &v in graph.neighbors(u) {
            let v = v as usize;
            let c_v = new_comm_ids[v] as usize;
            
            if c_u != c_v {
                *adj[c_u].entry(c_v).or_insert(0.0) += 1.0;
                total_weight += 1.0;
            } else {
                // Self-loop in super-graph?
                // Standard Louvain keeps self-loops for modularity calculation.
                *adj[c_u].entry(c_v).or_insert(0.0) += 1.0;
                total_weight += 1.0;
            }
        }
    }
    
    // 3. Convert to CompactGraph
    let mut offsets = Vec::with_capacity(num_new_nodes + 1);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    
    offsets.push(0);
    for i in 0..num_new_nodes {
        // Sort neighbors for deterministic output? Not strictly necessary but nice.
        let mut node_neighbors: Vec<_> = adj[i].iter().collect();
        node_neighbors.sort_by_key(|(&k, _)| k);
        
        for (&target, &weight) in node_neighbors {
            neighbors.push(target as u32);
            weights.push(weight);
        }
        offsets.push(neighbors.len());
    }
    
    let coarse_graph = CompactGraph {
        offsets,
        neighbors,
        weights: Some(weights),
        total_edge_weight: total_weight,
        num_nodes: num_new_nodes,
    };
    
    // Initial partition for next pass: each super-node is its own community
    let coarse_partition = (0..num_new_nodes as u32).collect();
    
    (coarse_graph, coarse_partition, new_comm_ids)
}
