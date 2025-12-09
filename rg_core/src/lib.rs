pub mod quadtree;
pub mod layout;
pub mod leiden;
pub use quadtree::{QuadTree, AABB};
pub use layout::{FruchtermanReingold, LayoutState, ForceAtlas2};

use glam::Vec2;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NodeData {
    pub position: Vec2,
    pub radius: f32,
    pub color: [f32; 4],
}

impl Default for NodeData {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            radius: 5.0,
            color: [1.0; 4],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub label: String,
    pub data: NodeData,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub id: usize,
    pub source: usize,
    pub target: usize,
}

#[derive(Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub adj: Vec<Vec<usize>>, // Adjacency list: node_index -> list of edge_indices
    pub adj_neighbors: Vec<Vec<usize>>, // Adjacency list: node_index -> list of neighbor_indices
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, position: Vec2, label: String) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            label,
            data: NodeData {
                position,
                radius: 5.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
        });
        self.adj.push(Vec::new());
        self.adj_neighbors.push(Vec::new());
        id
    }

    pub fn add_edge(&mut self, source: usize, target: usize) -> usize {
        let id = self.edges.len();
        self.edges.push(Edge { id, source, target });
        
        if source < self.adj.len() {
            self.adj[source].push(id);
            if source != target {
                self.adj_neighbors[source].push(target);
            }
        }
        if target < self.adj.len() && source != target {
            self.adj[target].push(id);
            self.adj_neighbors[target].push(source);
        }
        
        id
    }

    pub fn neighbors(&self, node_id: usize) -> &[usize] {
        if node_id >= self.adj_neighbors.len() {
            return &[];
        }
        &self.adj_neighbors[node_id]
    }
    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let mut graph = Graph::new();
        let n1 = graph.add_node(Vec2::new(0.0, 0.0));
        let n2 = graph.add_node(Vec2::new(10.0, 10.0));
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(n1, 0);
        assert_eq!(n2, 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::new();
        let n1 = graph.add_node(Vec2::ZERO);
        let n2 = graph.add_node(Vec2::ONE);
        let e1 = graph.add_edge(n1, n2);
        
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.edges[e1].source, n1);
        assert_eq!(graph.edges[e1].target, n2);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new();
        let n1 = graph.add_node(Vec2::ZERO);
        let n2 = graph.add_node(Vec2::ONE);
        let n3 = graph.add_node(Vec2::X);
        
        graph.add_edge(n1, n2);
        graph.add_edge(n1, n3);
        
        let neighbors = graph.neighbors(n1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n2));
        assert!(neighbors.contains(&n3));
        
        let n2_neighbors = graph.neighbors(n2);
        assert_eq!(n2_neighbors.len(), 1);
        assert!(n2_neighbors.contains(&n1));
    }
}
