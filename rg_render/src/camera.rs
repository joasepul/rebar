use glam::{Vec2, Mat4, Vec3};

#[derive(Debug)]
pub struct Camera {
    pub position: Vec2,
    pub zoom: f32,
    pub aspect_ratio: f32,
    pub min_zoom: f32,
    pub max_zoom: f32,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            aspect_ratio,
            min_zoom: 0.00001,
            max_zoom: 1000.0,
        }
    }

    pub fn build_view_projection_matrix(&self) -> Mat4 {
        // Orthographic projection
        // Left, Right, Bottom, Top, Near, Far
        // We want the view height to be controlled by zoom.
        // Let's say zoom=1 means view height is 2.0 (-1 to 1).
        
        let half_height = 1.0 / self.zoom;
        let half_width = half_height * self.aspect_ratio;
        
        let projection = Mat4::orthographic_rh(
            -half_width, half_width, 
            -half_height, half_height, 
            -1.0, 1.0
        );
        
        let view = Mat4::look_at_rh(
            Vec3::new(self.position.x, self.position.y, 1.0), // Camera is at z=1
            Vec3::new(self.position.x, self.position.y, 0.0), // Looking at z=0
            Vec3::Y, // Up is Y
        );
        
        projection * view
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }
    
    pub fn pan(&mut self, delta: Vec2) {
        // Delta is in screen coordinates? Or world?
        // Usually we receive screen delta. We need to scale by zoom.
        // For now assume delta is in world units.
        self.position -= delta;
    }
    
    pub fn zoom_at(&mut self, factor: f32, center: Vec2) {
        let old_zoom = self.zoom;
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(self.min_zoom, self.max_zoom);
        
        // Adjust position to keep center fixed
        // center_world = position + center_offset / zoom
        // We want center_world to be same before and after
        // position_new + center_offset / zoom_new = position_old + center_offset / zoom_old
        // position_new = position_old + center_offset * (1/zoom_old - 1/zoom_new)
        
        // center is the world position we want to keep fixed? 
        // No, usually 'center' passed here is world coordinates.
        // Let's assume 'center' is the world position of the mouse.
        
        // The camera position is the center of the screen in world coordinates.
        // Let M be the mouse position in world coordinates.
        // Let C be the camera position (center of screen) in world coordinates.
        // Vector V = M - C.
        // When we zoom in by factor F, the view shrinks.
        // The vector V should scale by 1/F relative to the screen center?
        // Actually simpler:
        // M = C + Offset / Zoom
        // We want M to stay constant.
        // C_new + Offset / Zoom_new = C_old + Offset / Zoom_old
        // C_new = C_old + Offset * (1/Zoom_old - 1/Zoom_new)
        // Offset is the screen delta from center to mouse (in world units at zoom=1).
        // Wait, let's look at main.rs.
        // In main.rs, we calculate mouse_world_pos.
        // So we pass mouse_world_pos as `center`.
        
        // M = P + (M - P)
        // We want M to be at the same screen location.
        // Screen location S = (M - P) * Zoom.
        // S_new = S_old
        // (M - P_new) * Zoom_new = (M - P_old) * Zoom_old
        // M - P_new = (M - P_old) * (Zoom_old / Zoom_new)
        // P_new = M - (M - P_old) * (Zoom_old / Zoom_new)
        
        self.position = center - (center - self.position) * (old_zoom / self.zoom);
    }
}
