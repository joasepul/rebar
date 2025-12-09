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

    pub fn step(&mut self, graph: &Graph, state: &mut LayoutState, active_mask: Option<&[bool]>) {
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
        for (i, &pos) in state.positions.iter().enumerate() {
            if let Some(mask) = active_mask {
                if !mask[i] { continue; }
            }
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);
        }
        // Add some padding
        let bounds = AABB::new(min_pos - Vec2::splat(10.0), max_pos + Vec2::splat(10.0));
        let mut quadtree = QuadTree::new(bounds);
        for (i, &pos) in state.positions.iter().enumerate() {
            if let Some(mask) = active_mask {
                if !mask[i] { continue; }
            }
            quadtree.insert(pos, i);
        }
        quadtree.calculate_mass();
        
        // 2. Calculate forces
        use rayon::prelude::*;
        
        let displacements: Vec<Vec2> = (0..node_count).into_par_iter().map(|i| {
            if let Some(mask) = active_mask {
                if !mask[i] { return Vec2::ZERO; }
            }
            
            let mut disp = Vec2::ZERO;
            let pos = state.positions[i];
            
            // Repulsion (Barnes-Hut)
            let repulsion = quadtree.calculate_repulsion(pos, 0.7, self.k * self.k);
            disp += repulsion;
            
            // Attraction (Edges)
            // Iterate neighbors
            for &neighbor_idx in graph.neighbors(i) {
                if let Some(mask) = active_mask {
                    if !mask[neighbor_idx] { continue; }
                }
                
                let other_pos = state.positions[neighbor_idx];
                let delta = other_pos - pos;
                let dist = delta.length();
                if dist > 0.0 {
                    // F_att = d^2 / k
                    let force = delta.normalize() * (dist * dist / self.k);
                    // In FR, attraction pulls u towards v.
                    // Here we are node i. Neighbor pulls us.
                    disp += force;
                }
            }
            
            disp
        }).collect();
        
        // 3. Apply forces
        let (new_positions, new_temperatures): (Vec<Vec2>, Vec<f32>) = (0..node_count).into_par_iter().map(|i| {
            let mut pos = state.positions[i];
            let mut temp = state.temperatures[i];
            
            if let Some(mask) = active_mask {
                if !mask[i] { return (pos, temp); }
            }
            
            let disp = displacements[i];
            
            let dist = disp.length();
            if dist > 0.0 {
                // Use max of global temperature and local node temperature
                let limit = self.temperature.max(temp);
                let limited_dist = dist.min(limit);
                pos += disp.normalize() * limited_dist;
            }
            
            // Cool down local temperature
            temp *= self.cooling;
            
            (pos, temp)
        }).unzip();
        
        state.positions = new_positions;
        state.temperatures = new_temperatures;
        
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
    pub fn new(scaling: f32) -> Self {
        Self {
            iterations: 0,
            gravity: 1.0,
            scaling, // Use the passed argument
            speed: 1.0,
            cooling: 0.99, // Slow cooling
            prevent_overlap: false,
            node_radius: 1.0,
            scale_by_degree: false,
            jitter_tolerance: 1.0,
        }
    }

    pub fn step(&mut self, graph: &Graph, state: &mut LayoutState, active_mask: Option<&[bool]>) {
        let node_count = graph.node_count();
        if node_count == 0 { return; }

        // 1. Build QuadTree for Barnes-Hut
        let mut min_pos = Vec2::splat(f32::MAX);
        let mut max_pos = Vec2::splat(f32::MIN);
        for (i, &pos) in state.positions.iter().enumerate() {
            if let Some(mask) = active_mask {
                if !mask[i] { continue; }
            }
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);
        }
        let bounds = AABB::new(min_pos - Vec2::splat(10.0), max_pos + Vec2::splat(10.0));
        let mut quadtree = QuadTree::new(bounds);
        for (i, &pos) in state.positions.iter().enumerate() {
            if let Some(mask) = active_mask {
                if !mask[i] { continue; }
            }
            quadtree.insert(pos, i);
        }
        quadtree.calculate_mass();

        // 2. Repulsion (Barnes-Hut) & 3. Gravity & 4. Attraction
        // We calculate all forces in parallel
        
        use rayon::prelude::*;
        
        let forces: Vec<Vec2> = (0..node_count).into_par_iter().map(|i| {
            if let Some(mask) = active_mask {
                if !mask[i] { return Vec2::ZERO; }
            }
            
            let mut force = Vec2::ZERO;
            let pos = state.positions[i];
            
            // Repulsion
            let repulsion = quadtree.calculate_repulsion(pos, 0.7, self.scaling * self.scaling);
            let degree = graph.neighbors(i).len() as f32;
            force += repulsion * (degree + 1.0) * 0.5;
            
            // Gravity
            let dist = pos.length();
            if dist > 0.0 {
                force -= pos.normalize() * (dist * self.gravity * 1.0);
            }
            
            // Attraction
            for &neighbor_idx in graph.neighbors(i) {
                if let Some(mask) = active_mask {
                    if !mask[neighbor_idx] { continue; }
                }
                
                let other_pos = state.positions[neighbor_idx];
                let delta = other_pos - pos;
                let dist = delta.length();
                if dist > 0.0 {
                    // Linear attraction: F = dist
                    // Direction is towards neighbor (delta)
                    force += delta.normalize() * dist;
                }
            }
            
            force
        }).collect();

        // 5. Apply Forces (Global Speed)
        // 5. Apply Forces (Adaptive Local Speed)
        // 5. Apply Forces (Adaptive Local Speed)
        // We need to update positions, old_forces.
        // state.positions, state.old_forces are Vecs.
        // We can use par_iter_mut on a zipped iterator or just iterate indices if we had a parallel zip mut.
        // Rayon supports par_iter_mut on slices.
        
        // We need to access forces[i] (read), old_forces[i] (read/write), positions[i] (read/write), graph.neighbors(i) (read).
        // Since we need random access to graph neighbors, maybe just par_iter over indices and use UnsafeCell or split slices?
        // Actually, we can just collect the new positions and old_forces and then replace?
        // Or use `par_iter_mut` on `state.positions` zipped with `state.old_forces` and `forces`.
        
        // Let's use a parallel iterator over the range and update a new vector of positions, then swap.
        // Or better, use `par_iter_mut` on a struct if we can?
        // Simplest: Compute new positions and new old_forces in parallel, then replace.
        
        let (new_positions, new_old_forces): (Vec<Vec2>, Vec<Vec2>) = (0..node_count).into_par_iter().map(|i| {
            let mut pos = state.positions[i];
            let mut force = forces[i]; // Default force
            
            if let Some(mask) = active_mask {
                if !mask[i] { 
                    // Return current pos and zero force (or old force?)
                    // If we return zero force, it resets momentum.
                    return (pos, Vec2::ZERO); 
                }
            }
            
            force = forces[i];
            let old_force = state.old_forces[i];
            
            // Swinging: How much the force vector has changed direction
            // Traction: How much the force vector is consistent
            let swinging = (force - old_force).length();
            let traction = (force + old_force).length();
            
            // Adaptive speed
            let jt = self.jitter_tolerance;
            let node_speed = 0.1 * self.speed * (jt * traction) / (jt * traction + swinging + 1e-6);
            
            // Mass-based integration
            let degree = graph.neighbors(i).len() as f32;
            let mass = degree + 1.0;
            
            let displacement = (force / mass) * node_speed;
            
            let dist = displacement.length();
            let max_displacement = 100.0;
            
            if dist > max_displacement {
                pos += displacement * (max_displacement / dist);
            } else {
                pos += displacement;
            }
            
            (pos, force)
        }).unzip();
        
        state.positions = new_positions;
        state.old_forces = new_old_forces;
        
        // 6. Prevent Overlap (Collision)
        if self.prevent_overlap {
            
            for i in 0..node_count {
                let pos = state.positions[i];
                
                // Calculate radius for node i
                let mut radius_i = graph.nodes[i].data.radius * self.node_radius;
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
                    let mut radius_j = graph.nodes[j].data.radius * self.node_radius;
                    if self.scale_by_degree {
                        let degree = graph.neighbors(j).len() as f32;
                        radius_j *= (degree + 1.0).sqrt();
                    }
                    
                    let min_dist = radius_i + radius_j;
                    
                    if dist_sq < min_dist * min_dist && dist_sq > 0.001 {
                        let dist = dist_sq.sqrt();
                        let overlap = min_dist - dist;
                        // Soft constraint: move each node by half the overlap
                        // This prevents overshoot and jitter
                        let correction = delta.normalize() * overlap * 0.5;
                        state.positions[i] += correction; 
                    }
                }
            }
        }
        
        // Cool down
        self.speed *= self.cooling;
        self.iterations += 1;
    }
}
