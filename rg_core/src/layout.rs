use glam::Vec2;
use crate::{Graph, QuadTree, AABB};

pub struct LayoutState {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub temperatures: Vec<f32>,
    pub old_forces: Vec<Vec2>,
}

impl LayoutState {
    pub fn new(node_count: usize) -> Self {
        Self {
            positions: vec![Vec2::ZERO; node_count],
            velocities: vec![Vec2::ZERO; node_count],
            temperatures: vec![0.0; node_count],
            old_forces: vec![Vec2::ZERO; node_count],
        }
    }
}

pub struct FruchtermanReingold {
    pub iterations: usize,
    pub temperature: f32,
    pub cooling: f32,
    pub k: f32, // Optimal distance
}

impl FruchtermanReingold {
    pub fn new(k: f32) -> Self {
        Self {
            iterations: 0,
            temperature: k * 10.0,
            cooling: 0.95,
            k,
        }
    }

    pub fn step(&mut self, graph: &Graph, state: &mut LayoutState) {
        let node_count = graph.node_count();
        if node_count == 0 {
            return;
        }
        
        // Repulsive forces
        // Naive O(N^2) for now. Optimization: Barnes-Hut using QuadTree.
        // For 50k nodes, O(N^2) is too slow (2.5 billion pairs).
        // We MUST use spatial index or just random sampling or limit to neighbors + random.
        // Or just implement Barnes-Hut later.
        // For Phase 1, let's do a simplified repulsion: only nearby nodes?
        let node_count = graph.nodes.len();
        if node_count == 0 { return; }
        
        // 1. Build QuadTree for Barnes-Hut
        // Find bounds
        let mut min_pos = Vec2::splat(f32::MAX);
        let mut max_pos = Vec2::splat(f32::MIN);
        for &pos in &state.positions {
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);
        }
        // Add some padding
        let bounds = AABB::new(min_pos - Vec2::splat(10.0), max_pos + Vec2::splat(10.0));
        let mut quadtree = QuadTree::new(bounds);
        for (i, &pos) in state.positions.iter().enumerate() {
            quadtree.insert(pos, i);
        }
        quadtree.calculate_mass();
        
        // 2. Calculate forces
        let mut displacements = vec![Vec2::ZERO; node_count];
        
        // Repulsion (Barnes-Hut)
        // Parallelize this? (Need rayon or crossbeam, but for now serial is fine to test correctness)
        for i in 0..node_count {
            let pos = state.positions[i];
            // Use QuadTree to calculate repulsion
            // We need to pass the repulsion constant (k^2 / d) logic into calculate_repulsion or adapt it.
            // The current calculate_repulsion uses a hardcoded strength. Let's make it configurable or pass it.
            // F_rep = k^2 / d
            // In calculate_repulsion: force += delta * (strength * mass / dist^2);
            // So strength should be k^2.
            
            let repulsion = quadtree.calculate_repulsion(pos, 0.7, self.k * self.k);
            
            displacements[i] += repulsion;
        }
        
        // Attraction (Edges)
        for edge in &graph.edges {
            let u = edge.source;
            let v = edge.target;
            let delta = state.positions[v] - state.positions[u];
            let dist = delta.length();
            if dist > 0.0 {
                // F_att = d^2 / k
                let force = delta.normalize() * (dist * dist / self.k);
                displacements[u] += force;
                displacements[v] -= force;
            }
        }
        
        // 3. Apply forces
        for i in 0..node_count {
            let disp = displacements[i];
            let dist = disp.length();
            if dist > 0.0 {
                // Use max of global temperature and local node temperature
                let limit = self.temperature.max(state.temperatures[i]);
                let limited_dist = dist.min(limit);
                state.positions[i] += disp.normalize() * limited_dist;
            }
            
            // Cool down local temperature
            state.temperatures[i] *= self.cooling;
        }
        
        // 4. Cool down
        self.temperature *= self.cooling;
        self.iterations += 1;
    }
}


pub struct ForceAtlas2 {
    pub iterations: usize,
    pub gravity: f32,
    pub scaling: f32, // Like 'k' but for FA2
    pub speed: f32,
    pub cooling: f32,
    pub prevent_overlap: bool,
    pub node_radius: f32,
    pub scale_by_degree: bool,
    pub jitter_tolerance: f32,
}

impl ForceAtlas2 {
    pub fn new(_scaling: f32) -> Self {
        Self {
            iterations: 0,
            gravity: 1.0,
            scaling: 10.0, // Reduced from scaling arg (usually 50.0) to prevent explosion
            speed: 1.0,
            cooling: 0.99, // Slow cooling
            prevent_overlap: false,
            node_radius: 1.0,
            scale_by_degree: false,
            jitter_tolerance: 1.0,
        }
    }

    pub fn step(&mut self, graph: &Graph, state: &mut LayoutState) {
        let node_count = graph.node_count();
        if node_count == 0 { return; }

        // 1. Build QuadTree for Barnes-Hut
        let mut min_pos = Vec2::splat(f32::MAX);
        let mut max_pos = Vec2::splat(f32::MIN);
        for &pos in &state.positions {
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);
        }
        let bounds = AABB::new(min_pos - Vec2::splat(10.0), max_pos + Vec2::splat(10.0));
        let mut quadtree = QuadTree::new(bounds);
        for (i, &pos) in state.positions.iter().enumerate() {
            quadtree.insert(pos, i);
        }
        quadtree.calculate_mass();

        let mut forces = vec![Vec2::ZERO; node_count];

        // 2. Repulsion (Barnes-Hut)
        // FA2 Repulsion: F = kr * (n1 + 1) * (n2 + 1) / d
        // We approximate mass as (degree + 1).
        // Our QuadTree currently stores mass as 1.0 per node.
        // We need to update QuadTree to support variable mass or just use uniform mass for now.
        // For standard FA2, mass is degree+1.
        // Let's stick to uniform mass for now to reuse existing QuadTree without major refactor,
        // or just assume mass=1 for repulsion which is "Dissuade Hubs" mode = false.
        
        for i in 0..node_count {
            let pos = state.positions[i];
            // FA2 uses linear repulsion in some modes, but standard is 1/d.
            // Our calculate_repulsion does k^2 / d.
            // Let's use it with scaling^2.
            let repulsion = quadtree.calculate_repulsion(pos, 0.7, self.scaling * self.scaling);
            
            // "Dissuade Hubs" approximation: Scale repulsion by degree.
            // This pushes hubs apart more strongly.
            let degree = graph.neighbors(i).len() as f32;
            forces[i] += repulsion * (degree + 1.0) * 0.5; 
        }

        // 3. Gravity
        for i in 0..node_count {
            let pos = state.positions[i];
            let dist = pos.length();
            if dist > 0.0 {
            // Stronger gravity for FA2 to keep it compact
            forces[i] -= pos.normalize() * (dist * self.gravity * 1.0); 
        }    }
        

        // 4. Attraction (Edges)
        // FA2 Attraction: F = d (linear)
        for edge in &graph.edges {
            let u = edge.source;
            let v = edge.target;
            let delta = state.positions[v] - state.positions[u];
            let dist = delta.length();
            if dist > 0.0 {
                // Linear attraction: F = dist
                let force = delta.normalize() * dist; 
                forces[u] += force;
                forces[v] -= force;
            }
        }

        // 5. Apply Forces (Global Speed)
        // 5. Apply Forces (Adaptive Local Speed)
        for i in 0..node_count {
            let force = forces[i];
            let old_force = state.old_forces[i];
            
            // Swinging: How much the force vector has changed direction
            // Traction: How much the force vector is consistent
            let swinging = (force - old_force).length();
            let traction = (force + old_force).length();
            
            // Adaptive speed
            // Global speed * (traction / (traction + swinging))
            // We add a small epsilon to avoid division by zero
            // Jitter tolerance allows tuning how aggressive the slowdown is
            let jt = self.jitter_tolerance;
            let mut node_speed = 0.1 * self.speed * (jt * traction) / (jt * traction + swinging + 1e-6);
            
            // Mass-based integration: F = ma => a = F/m
            // Mass = degree + 1
            let degree = graph.neighbors(i).len() as f32;
            let mass = degree + 1.0;
            
            // Apply displacement
            // displacement = (force / mass) * node_speed
            let displacement = (force / mass) * node_speed;
            
            // Cap displacement to prevent explosion (max 100.0 per step)
            let dist = displacement.length();
            let max_displacement = 100.0;
            
            if dist > max_displacement {
                state.positions[i] += displacement * (max_displacement / dist);
            } else {
                state.positions[i] += displacement;
            }
            
            // Store force for next step
            state.old_forces[i] = force;
        }
        
        // 6. Prevent Overlap (Collision)
        if self.prevent_overlap {
            let collision_force = 100.0; // Strong force
            
            for i in 0..node_count {
                let pos = state.positions[i];
                
                // Calculate radius for node i
                let mut radius_i = self.node_radius;
                if self.scale_by_degree {
                    let degree = graph.neighbors(i).len() as f32;
                    radius_i *= (degree + 1.0).sqrt();
                }
                
                // Query with max possible radius to be safe (simplification)
                // Or just use a large enough fixed query radius if we assume max degree isn't infinite.
                // Better: query with radius_i + max_possible_neighbor_radius?
                // For now, let's use a generous query radius.
                let query_radius = radius_i * 4.0; 
                
                let range = AABB::new(
                    pos - Vec2::splat(query_radius),
                    pos + Vec2::splat(query_radius)
                );
                
                let neighbors = quadtree.query(&range);
                for &j in &neighbors {
                    if i == j { continue; }
                    
                    let other_pos = state.positions[j];
                    let delta = pos - other_pos;
                    let dist_sq = delta.length_squared();
                    
                    // Calculate radius for node j
                    let mut radius_j = self.node_radius;
                    if self.scale_by_degree {
                        let degree = graph.neighbors(j).len() as f32;
                        radius_j *= (degree + 1.0).sqrt();
                    }
                    
                    let min_dist = radius_i + radius_j;
                    
                    if dist_sq < min_dist * min_dist && dist_sq > 0.001 {
                        let dist = dist_sq.sqrt();
                        let overlap = min_dist - dist;
                        let force = delta.normalize() * overlap * collision_force;
                        state.positions[i] += force * 0.1; 
                    }
                }
            }
        }
        
        // Cool down
        self.speed *= self.cooling;
        self.iterations += 1;
    }
}
