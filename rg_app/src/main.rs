use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
    keyboard::{KeyCode, PhysicalKey},
};
use rg_core::{Graph, QuadTree, AABB, FruchtermanReingold, LayoutState, ForceAtlas2, NodeData};
use rg_render::{Renderer, Camera};
use glam::Vec2;
use rand::Rng;
use std::sync::Arc;
use std::thread;
use crossbeam::channel::Receiver;
use std::collections::HashMap;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    graph: Graph,
    quadtree: QuadTree,
    rx: Receiver<LayoutUpdate>,
    control_tx: crossbeam::channel::Sender<LayoutParam>,
    current_k: f32,
    node_scale: f32,
    edge_scale: f32,
    scale_by_degree: bool,
    prevent_overlap: bool,
    leiden_gamma: f32,
    
    // Community State
    community_assignments: Vec<u32>,
    communities: HashMap<u32, CommunityInfo>,
    selected_tab: usize, // 0: Controls, 1: Communities
    hovered_community: Option<u32>,
    highlight_neighbors: bool,
    
    layout_mode: LayoutType,
    fa2_scaling: f32,
    fa2_gravity: f32,
    fr_temperature: f32,
    fr_cooling: f32,
    fr_k: f32,
    
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    
    // Interaction
    is_dragging_camera: bool,
    last_cursor_pos: Option<Vec2>,
    hovered_node: Option<usize>,
    selected_node: Option<usize>,
    is_dragging_node: bool,
    
    // Stats
    last_frame_time: std::time::Instant,
    frame_count: u32,
    fps: f32,
    node_count: usize,
}

impl App {
    fn new(rx: Receiver<LayoutUpdate>, control_tx: crossbeam::channel::Sender<LayoutParam>, graph: Graph, quadtree: QuadTree, node_count: usize) -> Self {
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(1.0), // Aspect ratio will be set on resize
            graph,
            quadtree,
            rx,
            control_tx,
            current_k: 50.0,
            node_scale: 1.0,
            edge_scale: 1.0,
            scale_by_degree: false,
            prevent_overlap: false,
            leiden_gamma: 1.0,
            community_assignments: Vec::new(),
            communities: HashMap::new(),
            selected_tab: 0,
            hovered_community: None,
            highlight_neighbors: true,
            layout_mode: LayoutType::FruchtermanReingold,
            fa2_scaling: 10.0,
            fa2_gravity: 1.0,
            fr_temperature: 100.0, // Default start temp
            fr_cooling: 0.95,
            fr_k: 10.0, // Default optimal distance
            egui_ctx: egui::Context::default(),
            egui_state: None,
            is_dragging_camera: false,
            last_cursor_pos: None,
            hovered_node: None,
            selected_node: None,
            is_dragging_node: false,
            last_frame_time: std::time::Instant::now(),
            frame_count: 0,
            fps: 0.0,
            node_count,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Rebar Graph Viewer");
            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());
            
            let mut renderer = pollster::block_on(Renderer::new(window.clone()));
            let size = window.inner_size();
            renderer.resize(size);
            self.renderer = Some(renderer);
            
            self.camera.resize(size.width, size.height);
            self.camera.zoom = 0.0005;
            
            self.egui_state = Some(egui_winit::State::new(
                self.egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &window,
                Some(window.scale_factor() as f32),
                None,
                None, // max_texture_side
            ));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = match self.window.as_ref() {
            Some(w) if w.id() == window_id => w,
            _ => return,
        };

        // Pass events to egui
        if let Some(egui_state) = &mut self.egui_state {
            let response = egui_state.on_window_event(window, &event);
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size);
                }
                self.camera.resize(physical_size.width, physical_size.height);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                // Check if egui wants the pointer
                if self.egui_ctx.wants_pointer_input() || self.egui_ctx.is_pointer_over_area() {
                    return;
                }

                match button {
                    MouseButton::Left => {
                        if state == ElementState::Pressed {
                            if let Some(node_idx) = self.hovered_node {
                                self.selected_node = Some(node_idx);
                                self.is_dragging_node = true;
                                // Pin the node in layout thread
                                let _ = self.control_tx.send(LayoutParam::DragNode(node_idx, self.graph.nodes[node_idx].data.position));
                            } else {
                                self.is_dragging_camera = true;
                                self.selected_node = None;
                            }
                        } else {
                            // Unpin the node
                            if self.is_dragging_node {
                                if let Some(idx) = self.selected_node {
                                    let _ = self.control_tx.send(LayoutParam::ReleaseNode(idx));
                                }
                            }
                            
                            self.is_dragging_camera = false;
                            self.is_dragging_node = false;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = Vec2::new(position.x as f32, position.y as f32);
                let size = window.inner_size();
                
                // Check if egui wants the pointer
                if self.egui_ctx.wants_pointer_input() || self.egui_ctx.is_pointer_over_area() {
                    self.hovered_node = None; // Clear hover if over UI
                    self.last_cursor_pos = Some(current_pos); // Still update cursor pos for zooming/panning if needed? No, probably not.
                    return;
                }
                
                // Convert screen pos to world pos
                let world_per_pixel = (2.0 / self.camera.zoom) / size.height as f32;
                
                // Screen center is (width/2, height/2)
                let screen_center = Vec2::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
                let screen_delta = current_pos - screen_center;
                
                // World pos relative to camera
                let world_delta = Vec2::new(screen_delta.x, -screen_delta.y) * world_per_pixel;
                let mouse_world_pos = self.camera.position + world_delta;
                
                // Find nearest node
                self.hovered_node = self.quadtree.find_nearest(mouse_world_pos, 15.0 * world_per_pixel);
                
                if self.is_dragging_node {
                    if let Some(idx) = self.selected_node {
                        self.graph.nodes[idx].data.position = mouse_world_pos;
                        // Update pinned position in layout thread
                        let _ = self.control_tx.send(LayoutParam::DragNode(idx, mouse_world_pos));
                    }
                } else if self.is_dragging_camera {
                    if let Some(last_pos) = self.last_cursor_pos {
                        let delta = current_pos - last_pos;
                        let world_delta = delta * world_per_pixel;
                        // To drag the world, we move camera in opposite direction of mouse
                        self.camera.pan(Vec2::new(world_delta.x, -world_delta.y)); 
                    }
                }
                self.last_cursor_pos = Some(current_pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.egui_ctx.wants_pointer_input() || self.egui_ctx.is_pointer_over_area() {
                    return;
                }
                
                let factor = match delta {
                    MouseScrollDelta::LineDelta(_, y) => 1.0 + y * 0.1,
                    MouseScrollDelta::PixelDelta(pos) => 1.0 + (pos.y as f32) * 0.001,
                };
                
                // Calculate current mouse world position for zoom center
                if let Some(cursor_pos) = self.last_cursor_pos {
                     let size = window.inner_size();
                     let world_per_pixel = (2.0 / self.camera.zoom) / size.height as f32;
                     let screen_center = Vec2::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
                     let screen_delta = cursor_pos - screen_center;
                     let world_delta = Vec2::new(screen_delta.x, -screen_delta.y) * world_per_pixel;
                     let mouse_world_pos = self.camera.position + world_delta;
                     
                     self.camera.zoom_at(factor, mouse_world_pos);
                } else {
                     self.camera.zoom_at(factor, self.camera.position);
                }
            }
            WindowEvent::RedrawRequested => {
                // FPS Calculation
                self.frame_count += 1;
                let now = std::time::Instant::now();
                if now.duration_since(self.last_frame_time).as_secs_f32() >= 1.0 {
                    self.frame_count = 0;
                    self.last_frame_time = now;
                }
                
                // Receive updates from layout thread
                while let Ok(update) = self.rx.try_recv() {
                    match update {
                        LayoutUpdate::Positions(positions) => {
                            for (i, &pos) in positions.iter().enumerate() {
                                if self.is_dragging_node {
                                    if let Some(dragged_idx) = self.selected_node {
                                        if i == dragged_idx { continue; }
                                    }
                                }
                                self.graph.nodes[i].data.position = pos;
                            }
                        }
                        LayoutUpdate::LeidenResult(assignments) => {
                            self.community_assignments = assignments.clone();
                            self.communities.clear();
                            
                            // Count members per community
                            let mut counts = HashMap::new();
                            for &comm in &self.community_assignments {
                                *counts.entry(comm).or_insert(0) += 1;
                            }
                            
                            // Assign unique colors (Golden Ratio)
                            let num_communities = counts.len();
                            let golden_ratio_conjugate = 0.618033988749895;
                            let mut h = 0.5; // Start at random hue
                            
                            // Sort communities by size for consistent coloring of largest ones?
                            // Or just iterate. Let's sort keys for determinism.
                            let mut comm_ids: Vec<u32> = counts.keys().cloned().collect();
                            comm_ids.sort();
                            
                            for comm_id in comm_ids {
                                h += golden_ratio_conjugate;
                                h %= 1.0;
                                let color = hsl_to_rgb(h * 360.0, 0.7, 0.6); // Saturation 0.7, Lightness 0.6
                                
                                self.communities.insert(comm_id, CommunityInfo {
                                    id: comm_id,
                                    color,
                                    count: *counts.get(&comm_id).unwrap(),
                                    visible: true,
                                });
                            }
                            
                            // Apply colors to nodes
                            for (i, &comm) in self.community_assignments.iter().enumerate() {
                                if let Some(info) = self.communities.get(&comm) {
                                    self.graph.nodes[i].data.color = info.color;
                                }
                            }
                            
                            // Switch to Communities tab
                            self.selected_tab = 1;
                        }
                    }
                }
                
                // Rebuild QuadTree
                let mut min_pos = Vec2::splat(f32::MAX);
                let mut max_pos = Vec2::splat(f32::MIN);
                for node in &self.graph.nodes {
                    min_pos = min_pos.min(node.data.position);
                    max_pos = max_pos.max(node.data.position);
                }
                
                // Add padding
                let bounds = AABB::new(min_pos - Vec2::splat(100.0), max_pos + Vec2::splat(100.0));
                self.quadtree = QuadTree::new(bounds);
                for (i, node) in self.graph.nodes.iter().enumerate() {
                    self.quadtree.insert(node.data.position, i);
                }
                self.quadtree.calculate_mass();
                
                if let Some(renderer) = &mut self.renderer {
                    renderer.update_camera(self.camera.build_view_projection_matrix());
                    
                    // Culling
                    let half_height = 1.0 / self.camera.zoom;
                    let half_width = half_height * self.camera.aspect_ratio;
                    let view_min = self.camera.position - Vec2::new(half_width, half_height);
                    let view_max = self.camera.position + Vec2::new(half_width, half_height);
                    let view_bounds = AABB::new(view_min, view_max);
                    
                    let visible_indices = self.quadtree.query(&view_bounds);
                    
                    let mut visible_nodes: Vec<NodeData> = visible_indices.iter()
                        .filter_map(|&i| {
                            // Check visibility
                            let mut is_dimmed = false;
                            
                            if !self.community_assignments.is_empty() {
                                if let Some(&comm_id) = self.community_assignments.get(i) {
                                    if let Some(info) = self.communities.get(&comm_id) {
                                        if !info.visible {
                                            return None;
                                        }
                                        
                                        // Check highlighting (Community)
                                        if let Some(hovered_id) = self.hovered_community {
                                            if comm_id != hovered_id {
                                                is_dimmed = true;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Check highlighting (Node Neighbors)
                            if !is_dimmed && self.highlight_neighbors {
                                if let Some(hovered_node_idx) = self.hovered_node {
                                    if i != hovered_node_idx {
                                        // Check if neighbor
                                        // This is O(N) inside the loop, making it O(N*VisibleN).
                                        // Ideally we should pre-calculate the set of highlighted nodes for the frame.
                                        // But for now, let's just check neighbors.
                                        // Optimization: `self.graph.neighbors(hovered_node_idx).contains(&i)` is O(Degree).
                                        // Since max degree is usually small relative to N, this is okay.
                                        // Actually `neighbors` returns a slice, so `contains` is linear scan of neighbors.
                                        let neighbors = self.graph.neighbors(hovered_node_idx);
                                        if !neighbors.contains(&i) {
                                            is_dimmed = true;
                                        }
                                    }
                                }
                            }
                            
                            let mut node = self.graph.nodes[i].data;
                            
                            // Apply scaling
                            if self.scale_by_degree {
                                let degree = self.graph.neighbors(i).len() as f32;
                                node.radius *= (degree + 1.0).sqrt(); // Sqrt to avoid massive nodes
                            }
                            node.radius *= self.node_scale;

                            if Some(i) == self.selected_node {
                                node.color = [1.0, 1.0, 0.0, 1.0];
                                node.radius *= 1.5;
                            } else if Some(i) == self.hovered_node {
                                node.color = [1.0, 0.5, 0.0, 1.0];
                                node.radius *= 1.2;
                            } else if is_dimmed {
                                // Dim non-highlighted nodes
                                node.color[3] = 0.1; // Low alpha
                                // Or darken color?
                                node.color[0] *= 0.2;
                                node.color[1] *= 0.2;
                                node.color[2] *= 0.2;
                            }
                            Some(node)
                        })
                        .collect();
                        
                    if visible_nodes.is_empty() {
                        visible_nodes.push(NodeData { position: Vec2::splat(-10000.0), radius: 0.0, color: [0.0; 4] });
                    }
                    
                    use rg_render::EdgeInstance;
                    let mut visible_edges = Vec::new();
                    
                    for edge in &self.graph.edges {
                        // Check visibility of endpoints
                        let mut is_dimmed = false;
                        
                        if !self.community_assignments.is_empty() {
                            let s_comm_opt = self.community_assignments.get(edge.source);
                            let t_comm_opt = self.community_assignments.get(edge.target);
                            
                            let s_visible = s_comm_opt.and_then(|c| self.communities.get(c)).map(|i| i.visible).unwrap_or(true);
                            let t_visible = t_comm_opt.and_then(|c| self.communities.get(c)).map(|i| i.visible).unwrap_or(true);
                            
                            if !s_visible || !t_visible {
                                continue;
                            }
                            
                            // Check highlighting (Community)
                            if let Some(hovered_id) = self.hovered_community {
                                let s_in_comm = s_comm_opt.map_or(false, |&c| c == hovered_id);
                                let t_in_comm = t_comm_opt.map_or(false, |&c| c == hovered_id);
                                
                                // Highlight if connected to the community
                                if !s_in_comm && !t_in_comm {
                                    is_dimmed = true;
                                }
                            }
                        }
                        
                        // Check highlighting (Node Neighbors)
                        if !is_dimmed && self.highlight_neighbors {
                            if let Some(hovered_node_idx) = self.hovered_node {
                                // Highlight only edges connected to the hovered node
                                if edge.source != hovered_node_idx && edge.target != hovered_node_idx {
                                    is_dimmed = true;
                                }
                            }
                        }

                        let source_pos = self.graph.nodes[edge.source].data.position;
                        let target_pos = self.graph.nodes[edge.target].data.position;
                        
                        if view_bounds.contains(source_pos) || view_bounds.contains(target_pos) {
                             visible_edges.push(EdgeInstance {
                                 start: source_pos.to_array(),
                                 end: target_pos.to_array(),
                                 color: if is_dimmed { [0.1, 0.1, 0.1, 0.02] } else { [0.5, 0.5, 0.5, 0.15] },
                                 width: self.edge_scale,
                             });
                        }
                    }
                        
                    renderer.update_instances(&visible_nodes);
                    renderer.update_edges(&visible_edges);

                    // GUI
                    if let Some(egui_state) = &mut self.egui_state {
                        let raw_input = egui_state.take_egui_input(window);
                        self.egui_ctx.begin_frame(raw_input);
                        
                        egui::Window::new("Rebar Stats").show(&self.egui_ctx, |ui| {
                            ui.horizontal(|ui| {
                            if ui.selectable_label(self.selected_tab == 0, "Layout").clicked() {
                                self.selected_tab = 0;
                            }
                            if ui.selectable_label(self.selected_tab == 1, "Communities").clicked() {
                                self.selected_tab = 1;
                            }
                            if ui.selectable_label(self.selected_tab == 2, "Visuals").clicked() {
                                self.selected_tab = 2;
                            }
                        });
                        ui.separator();

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            match self.selected_tab {
                                0 => {
                                    // LAYOUT TAB
                                    ui.label("Layout Algorithm:");
                                    if ui.radio_value(&mut self.layout_mode, LayoutType::FruchtermanReingold, "Fruchterman-Reingold").changed() {
                                        let _ = self.control_tx.send(LayoutParam::SwitchLayout(LayoutType::FruchtermanReingold));
                                    }
                                    if ui.radio_value(&mut self.layout_mode, LayoutType::ForceAtlas2, "ForceAtlas2").changed() {
                                        let _ = self.control_tx.send(LayoutParam::SwitchLayout(LayoutType::ForceAtlas2));
                                    }
                                    
                                    ui.separator();
                                    
                                    if self.layout_mode == LayoutType::ForceAtlas2 {
                                        ui.label("ForceAtlas2 Params:");
                                        if ui.add(egui::Slider::new(&mut self.fa2_scaling, 1.0..=100.0).text("Scaling")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFA2Scaling(self.fa2_scaling));
                                        }
                                        if ui.add(egui::Slider::new(&mut self.fa2_gravity, 0.1..=10.0).text("Gravity")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFA2Gravity(self.fa2_gravity));
                                        }
                                    } else {
                                        ui.label("Fruchterman-Reingold Params:");
                                        if ui.add(egui::Slider::new(&mut self.fr_k, 1.0..=100.0).text("Optimal Dist (K)")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFRK(self.fr_k));
                                        }
                                        if ui.add(egui::Slider::new(&mut self.fr_cooling, 0.90..=0.999).text("Cooling Rate")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFRCooling(self.fr_cooling));
                                        }
                                        if ui.add(egui::Slider::new(&mut self.fr_temperature, 0.1..=1000.0).text("Temperature")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFRTemperature(self.fr_temperature));
                                        }
                                    }
                                    
                                    ui.separator();
                                    if ui.checkbox(&mut self.prevent_overlap, "Prevent Overlap").changed() {
                                        let _ = self.control_tx.send(LayoutParam::PreventOverlap(self.prevent_overlap));
                                    }
                                    
                                    ui.separator();
                                    if ui.button("Recenter Camera").clicked() {
                                        // Use centroid (average position) instead of bounding box center
                                        // This is more robust against outliers.
                                        let mut sum_pos = Vec2::ZERO;
                                        let mut count = 0.0;
                                        
                                        let mut min_pos = Vec2::splat(f32::MAX);
                                        let mut max_pos = Vec2::splat(f32::MIN);
                                        
                                        for node in &self.graph.nodes {
                                            sum_pos += node.data.position;
                                            count += 1.0;
                                            
                                            min_pos = min_pos.min(node.data.position);
                                            max_pos = max_pos.max(node.data.position);
                                        }
                                        
                                        if count > 0.0 {
                                            let centroid = sum_pos / count;
                                            self.camera.position = centroid;
                                            
                                            // Calculate zoom to fit
                                            // We use the bounding box size, but maybe we should exclude outliers?
                                            // For now, bounding box is fine for zoom, but centering on centroid is key.
                                            let size = max_pos - min_pos;
                                            let max_dim = size.x.max(size.y).max(100.0);
                                            
                                            // 1.2 padding factor
                                            self.camera.zoom = 1.0 / (max_dim * 0.6); 
                                        }
                                    }
                                    
                                    ui.separator();
                                    ui.label("Interaction:");
                                    ui.label("  LMB + Drag: Pan Camera");
                                    ui.label("  Scroll: Zoom");
                                    ui.label("  Drag Node: Move Node");
                                }
                                1 => {
                                    // COMMUNITIES TAB
                                    ui.label("Community Detection:");
                                    ui.add(egui::Slider::new(&mut self.leiden_gamma, 0.1..=5.0).text("Resolution (Gamma)"));
                                    if ui.button("Detect Communities").clicked() {
                                        let _ = self.control_tx.send(LayoutParam::RunLeiden(self.leiden_gamma));
                                    }
                                    ui.separator();
                                    
                                    if self.communities.is_empty() {
                                        ui.label("No communities detected yet.");
                                    } else {
                                        ui.label(format!("Total Communities: {}", self.communities.len()));
                                        ui.separator();
                                        
                                        // Sort by size descending
                                        let mut comms: Vec<_> = self.communities.values_mut().collect();
                                        comms.sort_by(|a, b| b.count.cmp(&a.count));
                                        
                                        let mut changed = false;
                                        let mut ui_hovered_comm = None;
                                        
                                        for comm in comms {
                                            let response = ui.horizontal(|ui| {
                                                ui.checkbox(&mut comm.visible, "");
                                                let mut color = comm.color;
                                                if ui.color_edit_button_rgba_unmultiplied(&mut color).changed() {
                                                    comm.color = color;
                                                    changed = true;
                                                }
                                                ui.label(format!("ID: {} ({} nodes)", comm.id, comm.count));
                                            });
                                            
                                            if response.response.hovered() {
                                                ui_hovered_comm = Some(comm.id);
                                            }
                                        }
                                        
                                        // Update hovered community from UI
                                        if ui_hovered_comm.is_some() {
                                            self.hovered_community = ui_hovered_comm;
                                        } else if self.hovered_node.is_some() {
                                            // Update from Graph hover
                                            if let Some(node_idx) = self.hovered_node {
                                                if let Some(&comm_id) = self.community_assignments.get(node_idx) {
                                                    self.hovered_community = Some(comm_id);
                                                }
                                            }
                                        } else {
                                            self.hovered_community = None;
                                        }
                                        
                                        if changed {
                                            // Update all nodes
                                            for (i, &c) in self.community_assignments.iter().enumerate() {
                                                if let Some(info) = self.communities.get(&c) {
                                                    self.graph.nodes[i].data.color = info.color;
                                                }
                                            }
                                        }
                                    }
                                }
                                2 => {
                                    // VISUALS TAB
                                    ui.label("Rendering Settings:");
                                    if ui.add(egui::Slider::new(&mut self.node_scale, 0.1..=5.0).text("Node Scale")).changed() {
                                        let _ = self.control_tx.send(LayoutParam::NodeScale(self.node_scale));
                                    }
                                    // Edge scale is purely client-side rendering state, no need to send to layout thread?
                                    // Wait, edge_scale IS used in rendering loop, but is it used in layout?
                                    // Checking main.rs: edge_scale is used in rendering loop.
                                    // But previously it might have been sent?
                                    // Let's check LayoutParam. It has no EdgeScale variant.
                                    // So edge_scale is local to App.
                                    ui.add(egui::Slider::new(&mut self.edge_scale, 0.1..=5.0).text("Edge Scale"));
                                    
                                    if ui.checkbox(&mut self.scale_by_degree, "Scale Nodes by Degree").changed() {
                                        let _ = self.control_tx.send(LayoutParam::ScaleByDegree(self.scale_by_degree));
                                    }
                                    
                                    ui.separator();
                                    ui.label("Highlighting:");
                                    ui.checkbox(&mut self.highlight_neighbors, "Highlight Neighbors on Hover");
                                }
                                _ => {}
                            }
                        });
                        });
                        
                        let egui_output = self.egui_ctx.end_frame();
                        
                        egui_state.handle_platform_output(window, egui_output.platform_output);
                        
                        let clipped_primitives = self.egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);
                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [renderer.size.width, renderer.size.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        match renderer.render(Some((&clipped_primitives, &egui_output.textures_delta, &screen_descriptor))) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                }
            }
            _ => {}
    
    }
    }
    
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 4] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    [(r + m), (g + m), (b + m), 1.0]
}

fn load_graph_from_file(path: &str) -> std::io::Result<Graph> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    println!("Loading graph from {}...", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut edges = Vec::new();
    let mut max_id = 0;
    
    for line in reader.lines() {
        let line = line?;
        if line.trim().starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(u), Ok(v)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                edges.push((u, v));
                max_id = max_id.max(u).max(v);
            }
        }
    }
    
    let node_count = max_id + 1;
    println!("Graph has {} nodes and {} edges", node_count, edges.len());
    
    let mut graph = Graph::new();
    let mut rng = rand::thread_rng();
    
    // Initialize nodes with random positions
    for _ in 0..node_count {
        let pos = Vec2::new(
            rng.gen_range(-1000.0..1000.0),
            rng.gen_range(-1000.0..1000.0),
        );
        graph.add_node(pos);
    }
    
    for (u, v) in edges {
        graph.add_edge(u, v);
    }
    
    Ok(graph)
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let default_path = "facebook_combined.txt".to_string();
    let path = if args.len() > 1 {
        &args[1]
    } else {
        &default_path
    };
    
    // Load graph from file
    let graph = match load_graph_from_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load graph from {}: {}", path, e);
            if path != &default_path {
                 eprintln!("Trying default file: {}...", default_path);
                 match load_graph_from_file(&default_path) {
                     Ok(g) => g,
                     Err(e) => {
                         eprintln!("Failed to load default graph: {}", e);
                         eprintln!("Falling back to random graph...");
                         let mut graph = Graph::new();
                         // ... (random graph generation code is below, we can just let it fall through or duplicate logic)
                         // Actually the original code had a fallback block.
                         // Let's just return empty graph or random here to keep it simple.
                         Graph::new()
                     }
                 }
            } else {
                eprintln!("Falling back to random graph...");
                let mut graph = Graph::new();
                graph
            }
        }
    };
    
    // If graph is empty (failed load), generate random
    let graph = if graph.node_count() == 0 {
        let mut graph = Graph::new();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
             let pos = Vec2::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0));
             graph.add_node(pos);
        }
        graph
    } else {
        graph
    };

    
    println!("Building QuadTree...");
    let mut min_pos = Vec2::splat(f32::MAX);
    let mut max_pos = Vec2::splat(f32::MIN);
    for node in &graph.nodes {
        min_pos = min_pos.min(node.data.position);
        max_pos = max_pos.max(node.data.position);
    }
    // Add padding
    let bounds = AABB::new(min_pos - Vec2::splat(100.0), max_pos + Vec2::splat(100.0));
    
    let mut quadtree = QuadTree::new(bounds);
    for (i, node) in graph.nodes.iter().enumerate() {
        quadtree.insert(node.data.position, i);
    }
    quadtree.calculate_mass();
    
    // Layout Thread
    let graph_clone = graph.clone();
    let (tx, rx) = crossbeam::channel::unbounded::<LayoutUpdate>();
    
    // Layout Control Channel
    let (control_tx, control_rx) = crossbeam::channel::unbounded();
    
    thread::spawn(move || {
        enum ActiveLayout {
            FR(FruchtermanReingold),
            FA2(ForceAtlas2),
        }
        
        let mut active_layout = ActiveLayout::FR(FruchtermanReingold::new(50.0));
        let mut state = LayoutState::new(graph_clone.nodes.len());
        let mut pinned_nodes = std::collections::HashMap::new();
        
        for (i, node) in graph_clone.nodes.iter().enumerate() {
            state.positions[i] = node.data.position;
        }
        
        loop {
            // Check for control updates
            while let Ok(param) = control_rx.try_recv() {
                match param {
                    LayoutParam::K(k) => {
                        match &mut active_layout {
                            ActiveLayout::FR(l) => {
                                l.k = k;
                                l.temperature = k * 10.0;
                            }
                            ActiveLayout::FA2(l) => {
                                l.scaling = k;
                                l.speed = 1.0; // Wake up simulation
                            }
                        }
                    }
                    LayoutParam::DragNode(idx, pos) => {
                        pinned_nodes.insert(idx, pos);
                        // Heat up the dragged node and its neighbors gently
                        let wake_temp = match &mut active_layout {
                            ActiveLayout::FR(l) => l.k * 0.1,
                            ActiveLayout::FA2(l) => {
                                l.speed = l.speed.max(0.1); // Wake up slightly
                                l.scaling * 0.1
                            }
                        };
                        state.temperatures[idx] = wake_temp;
                        for &neighbor in graph_clone.neighbors(idx).iter() {
                            state.temperatures[neighbor] = wake_temp;
                        }
                    }
                    LayoutParam::ReleaseNode(idx) => {
                        pinned_nodes.remove(&idx);
                    }
                    LayoutParam::SwitchLayout(layout_type) => {
                        match layout_type {
                            LayoutType::FruchtermanReingold => {
                                active_layout = ActiveLayout::FR(FruchtermanReingold::new(50.0));
                            }
                            LayoutType::ForceAtlas2 => {
                                active_layout = ActiveLayout::FA2(ForceAtlas2::new(50.0));
                            }
                        }
                    }
                    LayoutParam::NodeScale(s) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.node_radius = s;
                        }
                    }
                    LayoutParam::PreventOverlap(b) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.prevent_overlap = b;
                        }
                    }
                    LayoutParam::ScaleByDegree(b) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.scale_by_degree = b;
                        }
                    }
                    LayoutParam::RunLeiden(gamma) => {
                        println!("Running Leiden with gamma={}", gamma);
                        let config = rg_core::leiden::LeidenConfig {
                            gamma,
                            max_passes: 10,
                            min_delta_q: 0.0001,
                            random_seed: 42,
                        };
                        let result = rg_core::leiden::leiden(&graph_clone, &config);
                        println!("Leiden finished: {} communities, Q={:.4}", result.num_communities, result.modularity);
                        
                        let _ = tx.send(LayoutUpdate::LeidenResult(result.community_of));
                    }
                    LayoutParam::UpdateFA2Scaling(s) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.scaling = s;
                        }
                    }
                    LayoutParam::UpdateFA2Gravity(g) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.gravity = g;
                        }
                    }
                    LayoutParam::UpdateFRTemperature(t) => {
                        if let ActiveLayout::FR(l) = &mut active_layout {
                            l.temperature = t;
                        }
                    }
                    LayoutParam::UpdateFRCooling(c) => {
                        if let ActiveLayout::FR(l) = &mut active_layout {
                            l.cooling = c;
                        }
                    }
                    LayoutParam::UpdateFRK(k) => {
                        if let ActiveLayout::FR(l) = &mut active_layout {
                            l.k = k;
                        }
                    }
                }
            }
            
            // Update pinned positions before step (so they affect repulsion)
            for (&idx, &pos) in &pinned_nodes {
                state.positions[idx] = pos;
                state.velocities[idx] = Vec2::ZERO;
            }
            
            match &mut active_layout {
                ActiveLayout::FR(l) => l.step(&graph_clone, &mut state),
                ActiveLayout::FA2(l) => l.step(&graph_clone, &mut state),
            }
            
            // Enforce pinned positions after step (so they don't move)
            for (&idx, &pos) in &pinned_nodes {
                state.positions[idx] = pos;
                state.velocities[idx] = Vec2::ZERO;
            }
            // Send positions to main thread
            let _ = tx.send(LayoutUpdate::Positions(state.positions.clone()));
            
            // Sleep to cap update rate (e.g. 60Hz)
            thread::sleep(std::time::Duration::from_millis(16));
        }
    });

    let node_count = graph.node_count();
    let mut app = App::new(rx, control_tx, graph, quadtree, node_count);
    event_loop.run_app(&mut app).unwrap();
}

enum LayoutParam {
    K(f32),
    DragNode(usize, Vec2),
    ReleaseNode(usize),
    SwitchLayout(LayoutType),
    NodeScale(f32),
    PreventOverlap(bool),
    ScaleByDegree(bool),
    RunLeiden(f32), // Gamma
    UpdateFA2Scaling(f32),
    UpdateFA2Gravity(f32),
    UpdateFRTemperature(f32),
    UpdateFRCooling(f32),
    UpdateFRK(f32),
}

struct CommunityInfo {
    id: u32,
    color: [f32; 4],
    count: usize,
    visible: bool,
}

enum LayoutUpdate {
    Positions(Vec<Vec2>),
    LeidenResult(Vec<u32>),
}

#[derive(Clone, Copy, PartialEq)]
enum LayoutType {
    FruchtermanReingold,
    ForceAtlas2,
}
