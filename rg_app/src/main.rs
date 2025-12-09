use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
    keyboard::{KeyCode, PhysicalKey},
};
use rg_core::{Graph, QuadTree, AABB, FruchtermanReingold, LayoutState, ForceAtlas2, NodeData};
use rg_render::{Renderer, Camera};
use glam::{Vec2, Vec3, Mat4};
use rand::Rng;
use std::sync::Arc;
use std::thread;
use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;

struct FilterState {
    // Ranges
    min_degree: f32,
    max_degree: f32,
    min_community_size: usize,
    max_community_size: usize,
    
    // Data limits
    data_min_degree: f32,
    data_max_degree: f32,
    data_max_community_size: usize,
    
    // Histograms
    degree_histogram: Vec<egui_plot::Bar>,
    community_size_histogram: Vec<egui_plot::Bar>,
    
    // Max counts for visualization
    max_degree_count: f64,
    max_community_count: f64,
    
    // Bin widths
    degree_bin_width: f32,
    community_bin_width: f32,
    
    // Dirty flags to recompute histograms
    dirty: bool,
    
    // Interaction state
    drag_start_degree: Option<f32>,
    drag_start_community: Option<f32>,
}

impl Default for FilterState {
    fn default() -> Self {
        Self {
            min_degree: 0.0,
            max_degree: f32::MAX,
            min_community_size: 0,
            max_community_size: usize::MAX,
            data_min_degree: 0.0,
            data_max_degree: 100.0,
            data_max_community_size: 100,
            degree_histogram: Vec::new(),
            community_size_histogram: Vec::new(),
            max_degree_count: 0.0,
            max_community_count: 0.0,
            degree_bin_width: 1.0,
            community_bin_width: 1.0,
            dirty: true,
            drag_start_degree: None,
            drag_start_community: None,
        }
    }
}

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
    text_scale: f32,
    scale_by_degree: bool,
    prevent_overlap: bool,
    leiden_gamma: f32,
    
    // Community State
    community_assignments: Vec<u32>,
    communities: HashMap<u32, CommunityInfo>,
    selected_tab: usize, // 0: Controls, 1: Communities
    hovered_community: Option<u32>,

    highlight_neighbors: bool,
    show_node_ids: bool,
    filter_state: FilterState,
    
    layout_mode: LayoutType,
    fa2_scaling: f32,
    fa2_gravity: f32,
    fr_temperature: f32,
    fr_cooling: f32,

    fr_k: f32,
    layout_running: bool,
    
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    
    // Interaction
    is_dragging_camera: bool,
    last_cursor_pos: Option<Vec2>,
    hovered_node: Option<usize>,
    selected_node: Option<usize>,
    dragging_node: Option<usize>,
    drag_start_pos: Option<Vec2>,
    
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
            text_scale: 1.0,
            scale_by_degree: false,
            prevent_overlap: false,
            leiden_gamma: 1.0,
            community_assignments: Vec::new(),
            communities: HashMap::new(),
            selected_tab: 0,
            hovered_community: None,

            highlight_neighbors: true,
            show_node_ids: false,
            filter_state: FilterState::default(),
            layout_mode: LayoutType::FruchtermanReingold,
            fa2_scaling: 10.0,
            fa2_gravity: 1.0,
            fr_temperature: 100.0, // Default start temp
            fr_cooling: 0.95,
            fr_k: 10.0, // Default optimal distance
            layout_running: true,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            is_dragging_camera: false,
            last_cursor_pos: None,
            hovered_node: None,
            selected_node: None,
            dragging_node: None,
            drag_start_pos: None,
            last_frame_time: std::time::Instant::now(),
            frame_count: 0,
            fps: 0.0,
            node_count,
        }
    }
    fn update_histograms(graph: &Graph, communities: &HashMap<u32, CommunityInfo>, filter_state: &mut FilterState) {
        if !filter_state.dirty {
            return;
        }
        
        // Degree Histogram
        let mut degrees = Vec::with_capacity(graph.node_count());
        let mut max_degree = 0.0f32;
        let mut min_degree = f32::MAX;
        
        for i in 0..graph.node_count() {
            let d = graph.neighbors(i).len() as f32;
            degrees.push(d);
            max_degree = max_degree.max(d);
            min_degree = min_degree.min(d);
        }
        
        if degrees.is_empty() {
            max_degree = 10.0;
            min_degree = 0.0;
        }
        
        filter_state.data_min_degree = min_degree;
        filter_state.data_max_degree = max_degree;
        
        // Initialize sliders if first run (max_degree was f32::MAX)
        if filter_state.max_degree == f32::MAX {
            filter_state.max_degree = max_degree;
        }
        
        // Binning
        let bin_count = 50;
        let bin_width = (max_degree - min_degree) / bin_count as f32;
        let bin_width = bin_width.max(1.0);
        filter_state.degree_bin_width = bin_width;
        
        let mut bins = vec![0u32; bin_count + 1];
        
        for &d in &degrees {
            let bin_idx = ((d - min_degree) / bin_width).floor() as usize;
            if bin_idx < bins.len() {
                bins[bin_idx] += 1;
            }
        }
        
        let mut max_count: f64 = 0.0;
        filter_state.degree_histogram = bins.iter().enumerate().map(|(i, &count)| {
            let x = min_degree + i as f32 * bin_width + bin_width * 0.5;
            max_count = max_count.max(count as f64);
            egui_plot::Bar::new(x as f64, count as f64).width(bin_width as f64)
        }).collect();
        filter_state.max_degree_count = max_count;
        
        // Community Size Histogram
        if !communities.is_empty() {
            let mut sizes = Vec::new();
            let mut max_size = 0;
            
            for comm in communities.values() {
                sizes.push(comm.count);
                max_size = max_size.max(comm.count);
            }
            
            filter_state.data_max_community_size = max_size;
            if filter_state.max_community_size == usize::MAX {
                filter_state.max_community_size = max_size;
            }
            
            let bin_count = 20;
            let bin_width = (max_size as f32 / bin_count as f32).max(1.0);
            filter_state.community_bin_width = bin_width;
            
            let mut bins = vec![0u32; bin_count + 1];
            for &s in &sizes {
                let bin_idx = ((s as f32) / bin_width).floor() as usize;
                if bin_idx < bins.len() {
                    bins[bin_idx] += 1;
                }
            }
            
            let mut max_count: f64 = 0.0;
            filter_state.community_size_histogram = bins.iter().enumerate().map(|(i, &count)| {
                let x = i as f32 * bin_width + bin_width * 0.5;
                max_count = max_count.max(count as f64);
                egui_plot::Bar::new(x as f64, count as f64).width(bin_width as f64)
            }).collect();
            filter_state.max_community_count = max_count;
        }
        
        filter_state.dirty = false;
    }
    
    fn update_layout_filters(graph: &Graph, communities: &HashMap<u32, CommunityInfo>, assignments: &[u32], filter_state: &FilterState, control_tx: &Sender<LayoutParam>) {
        // Compute active mask
        let mut active_mask = Vec::with_capacity(graph.node_count());
        
        for i in 0..graph.node_count() {
            let mut active = true;
            
            // Degree
            let degree = graph.neighbors(i).len() as f32;
            if degree < filter_state.min_degree || degree > filter_state.max_degree {
                active = false;
            }
            
            // Community
            if active && !assignments.is_empty() {
                if let Some(&comm_id) = assignments.get(i) {
                    if let Some(info) = communities.get(&comm_id) {
                        if info.count < filter_state.min_community_size || info.count > filter_state.max_community_size {
                            active = false;
                        }
                    }
                }
            }
            
            active_mask.push(active);
        }
        
        let _ = control_tx.send(LayoutParam::UpdateActiveNodes(active_mask));
    }

    fn draw_filter_plot(
        ui: &mut egui::Ui,
        id: &str,
        histogram: Vec<egui_plot::Bar>,
        height: f32,
        bar_color: egui::Color32,
        highlight_color: egui::Color32,
        current_range: std::ops::RangeInclusive<f32>,
        data_range: std::ops::RangeInclusive<f32>,
        max_count: f64,
        bin_width: f64,
        mut drag_start: Option<f32>,
        mut on_change: impl FnMut(f32, f32),
    ) -> Option<f32> {
        let chart = egui_plot::BarChart::new(histogram)
            .color(bar_color)
            .name(id);

        egui_plot::Plot::new(id)
            .height(height)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                let min = *current_range.start();
                let max = *current_range.end();
                
                // Highlight selection (behind bars)
                // Extend max by bin_width to cover the full bin
                let visual_max = max as f64 + bin_width;
                
                let highlight = egui_plot::Polygon::new(egui_plot::PlotPoints::new(vec![
                    [min as f64, 0.0],
                    [visual_max, 0.0],
                    [visual_max, max_count],
                    [min as f64, max_count],
                ]))
                .fill_color(highlight_color);
                plot_ui.polygon(highlight);
                
                // Vertical lines at bounds
                plot_ui.vline(egui_plot::VLine::new(min as f64).stroke(egui::Stroke::new(1.0, egui::Color32::WHITE)));
                plot_ui.vline(egui_plot::VLine::new(visual_max).stroke(egui::Stroke::new(1.0, egui::Color32::WHITE)));
                
                plot_ui.bar_chart(chart);
                
                // Handle interaction (Range Drag)
                if plot_ui.response().drag_started() {
                    if let Some(pointer_pos) = plot_ui.pointer_coordinate() {
                        drag_start = Some(pointer_pos.x as f32);
                    }
                }
                
                if plot_ui.response().dragged() {
                    if let Some(start) = drag_start {
                        if let Some(pointer_pos) = plot_ui.pointer_coordinate() {
                            let current = pointer_pos.x as f32;
                            let new_min = start.min(current).clamp(*data_range.start(), *data_range.end());
                            let new_max = start.max(current).clamp(*data_range.start(), *data_range.end());
                            on_change(new_min, new_max);
                        }
                    }
                }
                
                if plot_ui.response().drag_stopped() {
                    drag_start = None;
                }
            });
            
        drag_start
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
                            // Store start position for click detection
                            self.drag_start_pos = self.last_cursor_pos;
                            
                            if let Some(node_idx) = self.hovered_node {
                                // Start dragging node, but don't select yet
                                self.dragging_node = Some(node_idx);
                                // Pin the node in layout thread
                                let _ = self.control_tx.send(LayoutParam::DragNode(node_idx, self.graph.nodes[node_idx].data.position));
                            } else {
                                // Start dragging camera
                                self.is_dragging_camera = true;
                            }
                        } else {
                            // Released
                            let mut is_click = false;
                            if let (Some(start), Some(end)) = (self.drag_start_pos, self.last_cursor_pos) {
                                if (start - end).length() < 5.0 {
                                    is_click = true;
                                }
                            }
                            
                            // Handle Click (Selection)
                            if is_click {
                                if let Some(node_idx) = self.hovered_node {
                                    self.selected_node = Some(node_idx);
                                } else {
                                    self.selected_node = None;
                                }
                            }
                            
                            // Unpin the node if we were dragging
                            if let Some(idx) = self.dragging_node {
                                let _ = self.control_tx.send(LayoutParam::ReleaseNode(idx));
                            }
                            
                            self.is_dragging_camera = false;
                            self.dragging_node = None;
                            self.drag_start_pos = None;
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
                
                if let Some(idx) = self.dragging_node {
                    self.graph.nodes[idx].data.position = mouse_world_pos;
                    // Update pinned position in layout thread
                    let _ = self.control_tx.send(LayoutParam::DragNode(idx, mouse_world_pos));
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
                                if let Some(dragged_idx) = self.dragging_node {
                                    if i == dragged_idx { continue; }
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
                            self.filter_state.dirty = true;
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
                            
                            // 1. Degree Filter
                            // Optimization: We compute degree here. If we do it often, maybe cache it?
                            // But neighbors() is fast-ish (slice access).
                            let degree = self.graph.neighbors(i).len() as f32;
                            if degree < self.filter_state.min_degree || degree > self.filter_state.max_degree {
                                return None;
                            }
                            
                            // 2. Community Filter
                            if !self.community_assignments.is_empty() {
                                if let Some(&comm_id) = self.community_assignments.get(i) {
                                    if let Some(info) = self.communities.get(&comm_id) {
                                        if !info.visible {
                                            return None;
                                        }
                                        
                                        // Size filter
                                        if info.count < self.filter_state.min_community_size || info.count > self.filter_state.max_community_size {
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
                                if let Some(selected_idx) = self.selected_node {
                                    // Selection Mode:
                                    // 1. Highlight selected node and its neighbors
                                    // 2. Highlight the hovered node (but NOT its neighbors)
                                    let neighbors = self.graph.neighbors(selected_idx);
                                    let is_selected_group = i == selected_idx || neighbors.contains(&i);
                                    let is_hovered = Some(i) == self.hovered_node;
                                    
                                    if !is_selected_group && !is_hovered {
                                        is_dimmed = true;
                                    }
                                } else if let Some(hovered_idx) = self.hovered_node {
                                    // Hover Mode (No selection):
                                    // Highlight hovered node and its neighbors
                                    if i != hovered_idx {
                                        let neighbors = self.graph.neighbors(hovered_idx);
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
                        
                        // Check Degree Filter
                        let s_degree = self.graph.neighbors(edge.source).len() as f32;
                        let t_degree = self.graph.neighbors(edge.target).len() as f32;
                        
                        if s_degree < self.filter_state.min_degree || s_degree > self.filter_state.max_degree {
                            continue;
                        }
                        if t_degree < self.filter_state.min_degree || t_degree > self.filter_state.max_degree {
                            continue;
                        }

                        let mut s_comm_opt = None;
                        let mut t_comm_opt = None;
                        
                        if !self.community_assignments.is_empty() {
                            s_comm_opt = self.community_assignments.get(edge.source);
                            t_comm_opt = self.community_assignments.get(edge.target);
                            
                            // Check Community Visibility
                            let s_visible = s_comm_opt.and_then(|c| self.communities.get(c)).map(|i| i.visible).unwrap_or(true);
                            let t_visible = t_comm_opt.and_then(|c| self.communities.get(c)).map(|i| i.visible).unwrap_or(true);
                            
                            if !s_visible || !t_visible {
                                continue;
                            }
                            
                            // Check Community Size Filter
                             if let Some(&c) = s_comm_opt {
                                 if let Some(info) = self.communities.get(&c) {
                                     if info.count < self.filter_state.min_community_size || info.count > self.filter_state.max_community_size {
                                         continue;
                                     }
                                 }
                             }
                             if let Some(&c) = t_comm_opt {
                                 if let Some(info) = self.communities.get(&c) {
                                     if info.count < self.filter_state.min_community_size || info.count > self.filter_state.max_community_size {
                                         continue;
                                     }
                                 }
                             }
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

                        
                        // Check highlighting (Node Neighbors)
                        if !is_dimmed && self.highlight_neighbors {
                            if let Some(selected_idx) = self.selected_node {
                                // Selection Mode:
                                // Only highlight edges connected to the selected node
                                if edge.source != selected_idx && edge.target != selected_idx {
                                    is_dimmed = true;
                                }
                            } else if let Some(hovered_idx) = self.hovered_node {
                                // Hover Mode:
                                // Highlight edges connected to hovered node
                                if edge.source != hovered_idx && edge.target != hovered_idx {
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
                        
                        egui::Window::new("Rebar").show(&self.egui_ctx, |ui| {
                            // Selection UI
                            if let Some(selected_idx) = self.selected_node {
                                if let Some(node) = self.graph.nodes.get(selected_idx) {
                                    ui.group(|ui| {
                                        ui.heading("Selected Node");
                                        ui.label(format!("ID: {}", node.id));
                                        if !node.label.is_empty() {
                                            ui.label(format!("Label: {}", node.label));
                                        }
                                        ui.label(format!("Degree: {}", self.graph.neighbors(selected_idx).len()));
                                        if let Some(&comm_id) = self.community_assignments.get(selected_idx) {
                                            ui.label(format!("Community: {}", comm_id));
                                        }
                                    });
                                    ui.separator();
                                }
                            }

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
                            if ui.selectable_label(self.selected_tab == 3, "Filters").clicked() {
                                self.selected_tab = 3;
                            }
                        });
                        ui.separator();

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            match self.selected_tab {
                                0 => {
                                    // LAYOUT TAB
                                    ui.horizontal(|ui| {
                                        if self.layout_running {
                                            if ui.add(egui::Button::new("⏹ Stop Layout").fill(egui::Color32::from_rgb(200, 50, 50))).clicked() {
                                                self.layout_running = false;
                                                let _ = self.control_tx.send(LayoutParam::SetRunning(false));
                                            }
                                            ui.label("Running...");
                                        } else {
                                            if ui.add(egui::Button::new("▶ Run Layout").fill(egui::Color32::from_rgb(50, 200, 50))).clicked() {
                                                self.layout_running = true;
                                                let _ = self.control_tx.send(LayoutParam::SetRunning(true));
                                            }
                                            ui.label("Stopped");
                                        }
                                    });
                                    ui.separator();
                                    
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
                                        if ui.add(egui::Slider::new(&mut self.fa2_scaling, 1.0..=500.0).text("Scaling")).changed() {
                                            let _ = self.control_tx.send(LayoutParam::UpdateFA2Scaling(self.fa2_scaling));
                                        }
                                        if ui.add(egui::Slider::new(&mut self.fa2_gravity, 0.01..=10.0).text("Gravity")).changed() {
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
                                            let target_zoom = 1.0 / (max_dim * 0.6);
                                            
                                            // Adjust min_zoom to allow zooming out a bit more than fit
                                            self.camera.min_zoom = (target_zoom * 0.5).min(1e-5);
                                            self.camera.zoom = target_zoom;
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
                                    
                                    if ui.button("Reset Communities").clicked() {
                                        self.community_assignments.clear();
                                        self.communities.clear();
                                        self.hovered_community = None;
                                        self.filter_state.min_community_size = 0;
                                        self.filter_state.max_community_size = usize::MAX;
                                        
                                        // Reset node colors
                                        for node in &mut self.graph.nodes {
                                            node.data.color = [1.0, 1.0, 1.0, 1.0];
                                        }
                                        
                                        // Update filters (to clear any hiding)
                                        App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
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
                                                let is_filtered = comm.count < self.filter_state.min_community_size || comm.count > self.filter_state.max_community_size;
                                                
                                                let text = format!("ID: {} ({} nodes)", comm.id, comm.count);
                                                if is_filtered {
                                                    ui.label(egui::RichText::new(text).strikethrough().color(egui::Color32::GRAY));
                                                } else {
                                                    ui.label(text);
                                                }
                                            });
                                            
                                            if response.response.hovered() {
                                                ui_hovered_comm = Some(comm.id);
                                            }
                                        }
                                        
                                        // Update hovered community from UI
                                        // Update hovered community from UI
                                        if ui_hovered_comm.is_some() {
                                            self.hovered_community = ui_hovered_comm;
                                        } else {
                                            // Update from Graph hover or selection
                                            let source_node = self.hovered_node.or(self.selected_node);
                                            
                                            if let Some(node_idx) = source_node {
                                                if let Some(&comm_id) = self.community_assignments.get(node_idx) {
                                                    self.hovered_community = Some(comm_id);
                                                } else {
                                                    self.hovered_community = None;
                                                }
                                            } else {
                                                self.hovered_community = None;
                                            }
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
                                    ui.add(egui::Slider::new(&mut self.text_scale, 0.1..=5.0).text("Text Scale"));
                                    
                                    if ui.checkbox(&mut self.scale_by_degree, "Scale Nodes by Degree").changed() {
                                        let _ = self.control_tx.send(LayoutParam::ScaleByDegree(self.scale_by_degree));
                                    }
                                    
                                    ui.separator();
                                    ui.label("Highlighting:");
                                    ui.checkbox(&mut self.highlight_neighbors, "Highlight Neighbors on Hover");
                                    ui.checkbox(&mut self.show_node_ids, "Show Node ID on Hover");
                                }
                                
                                3 => {
                                    // FILTERS TAB
                                    App::update_histograms(&self.graph, &self.communities, &mut self.filter_state);
                                    
                                    ui.heading("Node Filters");
                                    ui.label("Filter nodes to hide them and exclude from layout.");
                                    ui.separator();


                                    let highlight_color = egui::Color32::from_rgba_unmultiplied(200, 200, 250, 10);
                                    let chart_height = 50.0;

                                    
                                    // Degree Filter
                                    ui.label(format!("Degree Range: {:.0} - {:.0}", self.filter_state.min_degree, self.filter_state.max_degree));
                                    
                                    ui.horizontal(|ui| {
                                        if ui.add(egui::DragValue::new(&mut self.filter_state.min_degree).range(self.filter_state.data_min_degree..=self.filter_state.max_degree).speed(1.0)).changed() {
                                            App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                        }
                                        ui.label("Min");
                                        if ui.add(egui::DragValue::new(&mut self.filter_state.max_degree).range(self.filter_state.min_degree..=self.filter_state.data_max_degree).speed(1.0)).changed() {
                                            App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                        }
                                        ui.label("Max");
                                        if ui.button("Reset").clicked() {
                                            self.filter_state.min_degree = self.filter_state.data_min_degree;
                                            self.filter_state.max_degree = self.filter_state.data_max_degree;
                                            App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                        }
                                    });
                                    
                                    self.filter_state.drag_start_degree = Self::draw_filter_plot(
                                        ui,
                                        "degree_hist",
                                        self.filter_state.degree_histogram.clone(),
                                        chart_height,
                                        egui::Color32::LIGHT_BLUE,
                                        highlight_color,
                                        self.filter_state.min_degree..=self.filter_state.max_degree,
                                        self.filter_state.data_min_degree..=self.filter_state.data_max_degree,
                                        self.filter_state.max_degree_count,
                                        self.filter_state.degree_bin_width as f64,
                                        self.filter_state.drag_start_degree,
                                        |min, max| {
                                            self.filter_state.min_degree = min;
                                            self.filter_state.max_degree = max;
                                            App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                        }
                                    );
                                        
                                    ui.separator();
                                    
                                    // Community Size Filter
                                    if !self.communities.is_empty() {
                                        ui.label(format!("Community Size Range: {} - {}", self.filter_state.min_community_size, self.filter_state.max_community_size));
                                        
                                        ui.horizontal(|ui| {
                                            if ui.add(egui::DragValue::new(&mut self.filter_state.min_community_size).range(0..=self.filter_state.max_community_size).speed(1.0)).changed() {
                                                App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                            }
                                            ui.label("Min");
                                            if ui.add(egui::DragValue::new(&mut self.filter_state.max_community_size).range(self.filter_state.min_community_size..=self.filter_state.data_max_community_size).speed(1.0)).changed() {
                                                App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                            }
                                            ui.label("Max");
                                            if ui.button("Reset").clicked() {
                                                self.filter_state.min_community_size = 0;
                                                self.filter_state.max_community_size = self.filter_state.data_max_community_size;
                                                App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                            }
                                        });
                                        
                                        self.filter_state.drag_start_community = Self::draw_filter_plot(
                                            ui,
                                            "comm_hist",
                                            self.filter_state.community_size_histogram.clone(),
                                            chart_height,
                                            egui::Color32::LIGHT_GREEN,
                                            highlight_color,
                                            self.filter_state.min_community_size as f32..=self.filter_state.max_community_size as f32,
                                            0.0..=self.filter_state.data_max_community_size as f32,
                                            self.filter_state.max_community_count,
                                            self.filter_state.community_bin_width as f64,
                                            self.filter_state.drag_start_community,
                                            |min, max| {
                                                self.filter_state.min_community_size = min as usize;
                                                self.filter_state.max_community_size = max as usize;
                                                App::update_layout_filters(&self.graph, &self.communities, &self.community_assignments, &self.filter_state, &self.control_tx);
                                            }
                                        );
                                    } else {
                                        ui.label("Run Community Detection to filter by community size.");
                                    }
                                }
                                _ => {}
                            }
                        });
                        });
                        
                        // Draw Node IDs
                        if self.show_node_ids {
                            let painter = self.egui_ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("labels")));
                            let view_proj = self.camera.build_view_projection_matrix();
                            let screen_size = Vec2::new(renderer.size.width as f32, renderer.size.height as f32);
                            let scale_factor = window.scale_factor() as f32;
                            
                            // Define closure to draw label
                            let draw_label = |node_idx: usize, painter: &egui::Painter| {
                                if let Some(node) = self.graph.nodes.get(node_idx) {
                                    // Check visibility
                                    let degree = self.graph.neighbors(node_idx).len() as f32;
                                    if degree < self.filter_state.min_degree || degree > self.filter_state.max_degree {
                                        return;
                                    }
                                    
                                    if !self.community_assignments.is_empty() {
                                        if let Some(&comm_id) = self.community_assignments.get(node_idx) {
                                            if let Some(info) = self.communities.get(&comm_id) {
                                                if !info.visible {
                                                    return;
                                                }
                                                if info.count < self.filter_state.min_community_size || info.count > self.filter_state.max_community_size {
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                    
                                    // World -> NDC
                                    let pos_world = node.data.position;
                                    let pos_ndc = view_proj.project_point3(Vec3::new(pos_world.x, pos_world.y, 0.0));
                                    
                                    // Check if behind camera (though orthographic usually doesn't have this issue unless clipped)
                                    // NDC z is 0 to 1 (wgpu) or -1 to 1 (gl)? 
                                    // WGPU uses 0 to 1 for Z.
                                    // But project_point3 returns normalized coordinates.
                                    
                                    // NDC -> Screen
                                    // NDC x: -1 to 1 -> 0 to width
                                    // NDC y: -1 to 1 -> height to 0 (Y up in NDC, Y down in Screen)
                                    // Wait, egui Y is down. WGPU NDC Y is up.
                                    
                                    let x = (pos_ndc.x + 1.0) * 0.5 * screen_size.x;
                                    let y = (1.0 - pos_ndc.y) * 0.5 * screen_size.y;
                                    
                                    let screen_pos = egui::Pos2::new(x / scale_factor, y / scale_factor);
                                    
                                    let text = if node.label.is_empty() {
                                        format!("{}", node.id)
                                    } else {
                                        node.label.clone()
                                    };
                                    
                                    // Calculate font size
                                    // 1. Determine points per world unit
                                    // NDC height is 2.0 (-1 to 1)
                                    // Screen height in points is screen_size.y / scale_factor
                                    // 1 NDC unit = (screen_size.y / scale_factor) / 2.0 points
                                    // 1 World unit = camera.zoom NDC units
                                    let screen_height_points = screen_size.y / scale_factor;
                                    let points_per_world_unit = self.camera.zoom * screen_height_points * 0.5;
                                    
                                    let degree_scale = if self.scale_by_degree {
                                        (degree + 1.0).ln()
                                    } else {
                                        1.0
                                    };
                                    
                                    // Base font size in WORLD UNITS (relative to node radius of ~5.0)
                                    // Let's say we want text to be about 2.5x the node radius by default
                                    let base_size_world = 12.0; 
                                    
                                    let font_size = (base_size_world * points_per_world_unit * self.text_scale * self.node_scale * degree_scale).max(1.0);
                                    
                                    // Determine text color
                                    let mut text_color = egui::Color32::WHITE;
                                    if let Some(&comm_id) = self.community_assignments.get(node_idx) {
                                        if let Some(info) = self.communities.get(&comm_id) {
                                            let [r, g, b, _] = info.color;
                                            text_color = egui::Color32::from_rgb(
                                                (r * 255.0) as u8,
                                                (g * 255.0) as u8,
                                                (b * 255.0) as u8,
                                            );
                                        }
                                    }

                                    // Draw background rect for readability
                                    let galley = painter.layout_no_wrap(text, egui::FontId::proportional(font_size), text_color);
                                    let rect = galley.rect.translate(screen_pos - galley.rect.center_bottom());
                                    let padded_rect = rect.expand(2.0);
                                    
                                    painter.rect_filled(padded_rect, 2.0, egui::Color32::from_black_alpha(150));
                                    painter.galley(rect.min, galley, text_color);
                                }
                            };
                            
                            if let Some(selected_idx) = self.selected_node {
                                // Selection Mode
                                // Draw selected node and neighbors
                                for &neighbor in self.graph.neighbors(selected_idx) {
                                    draw_label(neighbor, &painter);
                                }
                                draw_label(selected_idx, &painter);
                                
                                // Also draw hovered node if it's not already drawn
                                if let Some(hovered_idx) = self.hovered_node {
                                    let neighbors = self.graph.neighbors(selected_idx);
                                    if hovered_idx != selected_idx && !neighbors.contains(&hovered_idx) {
                                        draw_label(hovered_idx, &painter);
                                    }
                                }
                            } else if let Some(hovered_idx) = self.hovered_node {
                                // Hover Mode
                                for &neighbor in self.graph.neighbors(hovered_idx) {
                                    draw_label(neighbor, &painter);
                                }
                                draw_label(hovered_idx, &painter);
                            }
                        }
                        
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

fn url_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next();
            let h2 = chars.next();
            if let (Some(h1), Some(h2)) = (h1, h2) {
                if let Ok(byte) = u8::from_str_radix(&format!("{}{}", h1, h2), 16) {
                    bytes.push(byte);
                    continue;
                }
                // If invalid hex, push the original chars
                bytes.push(b'%');
                let mut buf = [0; 4];
                for &b in h1.encode_utf8(&mut buf).as_bytes() { bytes.push(b); }
                for &b in h2.encode_utf8(&mut buf).as_bytes() { bytes.push(b); }
            } else {
                bytes.push(b'%');
                if let Some(h) = h1 { 
                    let mut buf = [0; 4];
                    for &b in h.encode_utf8(&mut buf).as_bytes() { bytes.push(b); }
                }
            }
        } else {
            let mut buf = [0; 4];
            for &b in c.encode_utf8(&mut buf).as_bytes() {
                bytes.push(b);
            }
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

fn load_graph_from_file(path: &str) -> std::io::Result<Graph> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::collections::HashMap;
    
    println!("Loading graph from {}...", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut edges = Vec::new();
    let mut node_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 0;
    
    // Helper to get or create node ID
    let mut get_id = |name: &str| -> usize {
        // Decode URL-encoded names (e.g. %20 -> space, %C3%A9 -> é)
        let decoded_name = url_decode(name);
        if let Some(&id) = node_map.get(&decoded_name) {
            id
        } else {
            let id = next_id;
            node_map.insert(decoded_name, id);
            next_id += 1;
            id
        }
    };
    
    for line in reader.lines() {
        let line = line?;
        if line.trim().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            // Try parsing as usize first (legacy support for numeric edge lists)
            // But if it fails, treat as string labels
            let u_str = parts[0];
            let v_str = parts[1];
            
            // Check if they look like numbers
            let u_is_num = u_str.parse::<usize>().is_ok();
            let v_is_num = v_str.parse::<usize>().is_ok();
            
            if u_is_num && v_is_num {
                 let u = u_str.parse::<usize>().unwrap();
                 let v = v_str.parse::<usize>().unwrap();
                 
                 // Ensure we map these numbers to our ID space if we are mixing types, 
                 // but for pure numeric files, we usually want to preserve IDs.
                 // However, to support mixed or sparse IDs, let's just treat them as labels too?
                 // Actually, if we treat "1" as a label, it gets mapped to 0.
                 // This is safer for sparse graphs.
                 let u_id = get_id(u_str);
                 let v_id = get_id(v_str);
                 edges.push((u_id, v_id));
            } else {
                // String labels (e.g. wiki-links.tsv)
                let u_id = get_id(u_str);
                let v_id = get_id(v_str);
                edges.push((u_id, v_id));
            }
        }
    }
    
    let node_count = next_id;
    println!("Graph has {} nodes and {} edges", node_count, edges.len());
    
    let mut graph = Graph::new();
    let mut rng = rand::thread_rng();
    
    // Create nodes in order of IDs (0 to node_count-1)
    // We need to reconstruct the label from the map.
    // Invert the map: ID -> Label
    let mut id_to_label = vec![String::new(); node_count];
    for (label, &id) in &node_map {
        id_to_label[id] = label.clone();
    }
    
    // Initialize nodes with random positions
    for i in 0..node_count {
        let pos = Vec2::new(
            rng.gen_range(-1000.0..1000.0),
            rng.gen_range(-1000.0..1000.0),
        );
        graph.add_node(pos, id_to_label[i].clone());
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
        for i in 0..100 {
             let pos = Vec2::new(rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0));
             graph.add_node(pos, format!("Node {}", i));
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
        
        let mut active_mask: Option<Vec<bool>> = None;
        let mut is_running = true;
        
        loop {
            // Check for control updates
            while let Ok(param) = control_rx.try_recv() {
                match param {
                    LayoutParam::UpdateActiveNodes(mask) => {
                        active_mask = Some(mask);
                    }
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
                                active_layout = ActiveLayout::FA2(ForceAtlas2::new(100.0));
                            }
                        }
                    }
                    LayoutParam::NodeScale(s) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.node_radius = s;
                            l.speed = 1.0; // Wake up
                        }
                    }
                    LayoutParam::PreventOverlap(b) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.prevent_overlap = b;
                            l.speed = 1.0; // Wake up
                        }
                    }
                    LayoutParam::ScaleByDegree(b) => {
                        if let ActiveLayout::FA2(l) = &mut active_layout {
                            l.scale_by_degree = b;
                            l.speed = 1.0; // Wake up
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
                    LayoutParam::SetRunning(running) => {
                        is_running = running;
                    }
                }
            }
            
            // Update pinned positions before step (so they affect repulsion)
            for (&idx, &pos) in &pinned_nodes {
                state.positions[idx] = pos;
                state.velocities[idx] = Vec2::ZERO;
            }
            
            if is_running {
                match &mut active_layout {
                    ActiveLayout::FR(l) => l.step(&graph_clone, &mut state, active_mask.as_deref()),
                    ActiveLayout::FA2(l) => l.step(&graph_clone, &mut state, active_mask.as_deref()),
                }
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
    SetRunning(bool),
    RunLeiden(f32), // Gamma
    UpdateFA2Scaling(f32),
    UpdateFA2Gravity(f32),
    UpdateFRTemperature(f32),
    UpdateFRCooling(f32),

    UpdateFRK(f32),
    UpdateActiveNodes(Vec<bool>),
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
