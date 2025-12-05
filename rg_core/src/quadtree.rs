use glam::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y
    }
}

pub struct QuadTree {
    pub nodes: Vec<QuadNode>,
    pub root: usize,
}

struct QuadNode {
    bounds: AABB,
    children: Option<[usize; 4]>, // Indices into nodes
    points: Vec<(Vec2, usize)>, // Position and data index
    center_of_mass: Vec2,
    total_mass: f32,
}

impl QuadTree {
    pub fn new(bounds: AABB) -> Self {
        let root = QuadNode {
            bounds,
            children: None,
            points: Vec::new(),
            center_of_mass: Vec2::ZERO,
            total_mass: 0.0,
        };
        Self {
            nodes: vec![root],
            root: 0,
        }
    }

    // ... (insert and subdivide need to initialize new nodes with zero mass)

    pub fn calculate_mass(&mut self) {
        self.calculate_mass_recursive(self.root);
    }

    fn calculate_mass_recursive(&mut self, node_idx: usize) -> (Vec2, f32) {
        // We need to borrow nodes mutably, but we can't do that easily with recursion on the same Vec.
        // We can use indices.
        
        // First, calculate mass from points in this node
        let mut center = Vec2::ZERO;
        let mut mass = 0.0;
        
        // Need to access self.nodes[node_idx], but we also need to call recursively.
        // To avoid borrow checker issues with recursive mutable borrow of `self.nodes`,
        // we can't easily do this in one pass if we store children indices.
        // Actually, we can, because we just need to read children indices, then recurse, then write back.
        
        // Clone children indices to avoid holding borrow
        let children = self.nodes[node_idx].children;
        let points = self.nodes[node_idx].points.clone(); // Clone points for now (optimization: avoid clone)
        
        for (pos, _) in &points {
            center += *pos;
            mass += 1.0; // Assume unit mass for nodes
        }
        
        if let Some(child_indices) = children {
            for &child_idx in &child_indices {
                let (child_center, child_mass) = self.calculate_mass_recursive(child_idx);
                center += child_center * child_mass;
                mass += child_mass;
            }
        }
        
        if mass > 0.0 {
            center /= mass;
        }
        
        // Write back
        self.nodes[node_idx].center_of_mass = center;
        self.nodes[node_idx].total_mass = mass;
        
        (center, mass)
    }
    
    // Barnes-Hut Force Calculation
    // theta: threshold (usually 0.5 or 0.7)
    // strength: repulsion constant (k^2)
    pub fn calculate_repulsion(&self, point: Vec2, theta: f32, strength: f32) -> Vec2 {
        let mut force = Vec2::ZERO;
        let mut stack = vec![self.root];
        
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            
            if node.total_mass == 0.0 {
                continue;
            }
            
            let delta = point - node.center_of_mass;
            let dist_sq = delta.length_squared();
            let dist = dist_sq.sqrt();
            
            // Width of the region
            let width = node.bounds.max.x - node.bounds.min.x;
            
            // If node is far enough (or is a leaf with 1 point which is not self), treat as single body
            // Condition: width / dist < theta
            // Also check if it's not the point itself (dist > epsilon)
            
            if width / dist < theta || node.children.is_none() {
                if dist > 0.1 { // Avoid self-repulsion and singularity
                    // F_rep = k^2 * mass / dist (Fruchterman-Reingold uses inverse distance, not inverse squared)
                    // force vector = dir * scalar_force
                    // dir = delta / dist
                    // scalar_force = strength * mass / dist
                    // force = (delta / dist) * (strength * mass / dist) = delta * strength * mass / dist^2
                    
                    force += delta * (strength * node.total_mass / dist_sq);
                }
            } else {
                // Too close, recurse
                if let Some(children) = node.children {
                    for &child_idx in &children {
                        stack.push(child_idx);
                    }
                }
            }
        }
        
        force
    }

    pub fn insert(&mut self, point: Vec2, data_index: usize) {
        self.insert_recursive(self.root, point, data_index, 0);
    }

    fn insert_recursive(&mut self, node_idx: usize, point: Vec2, data_index: usize, depth: u32) {
        if !self.nodes[node_idx].bounds.contains(point) {
            return;
        }

        if self.nodes[node_idx].children.is_none() && self.nodes[node_idx].points.len() >= 16 && depth < 10 {
            self.subdivide(node_idx, depth);
        }

        if let Some(children) = self.nodes[node_idx].children {
            for &child_idx in &children {
                if self.nodes[child_idx].bounds.contains(point) {
                    self.insert_recursive(child_idx, point, data_index, depth + 1);
                    return;
                }
            }
        } else {
            self.nodes[node_idx].points.push((point, data_index));
        }
    }

    fn subdivide(&mut self, node_idx: usize, depth: u32) {
        let bounds = self.nodes[node_idx].bounds;
        let center = (bounds.min + bounds.max) * 0.5;
        
        let quads = [
            AABB::new(Vec2::new(bounds.min.x, center.y), Vec2::new(center.x, bounds.max.y)), // TL
            AABB::new(center, bounds.max), // TR
            AABB::new(bounds.min, center), // BL
            AABB::new(Vec2::new(center.x, bounds.min.y), Vec2::new(bounds.max.x, center.y)), // BR
        ];

        let mut children_indices = [0; 4];
        for (i, quad) in quads.iter().enumerate() {
            let child = QuadNode {
                bounds: *quad,
                children: None,
                points: Vec::new(),
                center_of_mass: Vec2::ZERO,
                total_mass: 0.0,
            };
            let idx = self.nodes.len();
            self.nodes.push(child);
            children_indices[i] = idx;
        }
        
        self.nodes[node_idx].children = Some(children_indices);
        
        let points = std::mem::take(&mut self.nodes[node_idx].points);
        for (pos, data) in points {
            for &child_idx in &children_indices {
                if self.nodes[child_idx].bounds.contains(pos) {
                    self.insert_recursive(child_idx, pos, data, depth + 1);
                    break;
                }
            }
        }
    }

    pub fn query(&self, range: &AABB) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![self.root];
        
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            
            if !node.bounds.intersects(range) {
                continue;
            }

            for &(pos, data) in &node.points {
                if range.contains(pos) {
                    result.push(data);
                }
            }

            if let Some(children) = node.children {
                for &child_idx in &children {
                    stack.push(child_idx);
                }
            }
        }
        result
    }

    pub fn depth(&self) -> usize {
        self.depth_recursive(self.root)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn depth_recursive(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        if let Some(children) = node.children {
            1 + children.iter().map(|&c| self.depth_recursive(c)).max().unwrap_or(0)
        } else {
            1
        }
    }
    pub fn find_nearest(&self, point: Vec2, max_radius: f32) -> Option<usize> {
        let mut best_dist_sq = max_radius * max_radius;
        let mut best_idx = None;
        
        let mut stack = vec![self.root];
        
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            
            // Optimization: check if node bounds are within best_dist of point
            // Distance from point to AABB
            let dx = (node.bounds.min.x - point.x).max(0.0).max(point.x - node.bounds.max.x);
            let dy = (node.bounds.min.y - point.y).max(0.0).max(point.y - node.bounds.max.y);
            let dist_sq = dx * dx + dy * dy;
            
            if dist_sq > best_dist_sq {
                continue;
            }
            
            for &(pos, data) in &node.points {
                let d2 = pos.distance_squared(point);
                if d2 < best_dist_sq {
                    best_dist_sq = d2;
                    best_idx = Some(data);
                }
            }
            
            if let Some(children) = node.children {
                // Optimization: sort children by distance to point?
                for &child_idx in &children {
                    stack.push(child_idx);
                }
            }
        }
        
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadtree_insert_query() {
        let bounds = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
        let mut qt = QuadTree::new(bounds);

        qt.insert(Vec2::new(10.0, 10.0), 1);
        qt.insert(Vec2::new(90.0, 90.0), 2);
        qt.insert(Vec2::new(50.0, 50.0), 3);

        let query_bounds = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(20.0, 20.0));
        let results = qt.query(&query_bounds);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1);
        
        let query_all = AABB::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
        let results_all = qt.query(&query_all);
        assert_eq!(results_all.len(), 3);
    }
    
    #[test]
    fn test_subdivision() {
        let bounds = AABB::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let mut qt = QuadTree::new(bounds);
        
        for i in 0..20 {
            qt.insert(Vec2::new(i as f32, i as f32), i);
        }
        
        // Root should have children now
        assert!(qt.nodes[qt.root].children.is_some());
        
        let query_bounds = AABB::new(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let results = qt.query(&query_bounds);
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn test_find_nearest() {
        let bounds = AABB::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let mut qt = QuadTree::new(bounds);
        
        qt.insert(Vec2::new(10.0, 10.0), 1);
        qt.insert(Vec2::new(20.0, 20.0), 2);
        
        let nearest = qt.find_nearest(Vec2::new(12.0, 12.0), 5.0);
        assert_eq!(nearest, Some(1));
        
        let nearest_none = qt.find_nearest(Vec2::new(50.0, 50.0), 5.0);
        assert_eq!(nearest_none, None);
    }
}
