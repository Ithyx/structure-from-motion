use std::path::Path;

use morrigu::bevy_ecs::prelude::Entity;
use morrigu::{
    allocated_types::AllocatedBuffer,
    application::{ApplicationState, BuildableApplicationState},
    components::{
        camera::{Camera, PerspectiveData},
        mesh_rendering::default_ubo_bindings,
        resource_wrapper::ResourceWrapper,
        transform::Transform,
    },
    descriptor_resources::DescriptorResources,
    math_types::{Quat, Vec2, Vec3},
    shader::Shader,
    systems::mesh_renderer,
    utils::ThreadSafeRef,
    vertices::simple::SimpleVertex,
};
use morrigu::{egui, winit};

use crate::camera::ViewerCamera;

type Vertex = SimpleVertex;
type Material = morrigu::material::Material<Vertex>;
type Mesh = morrigu::mesh::Mesh<Vertex>;
type MeshRendering = morrigu::components::mesh_rendering::MeshRendering<Vertex>;

pub struct Point {
    pub position: Vec3,
    pub color: Vec3,
}

pub struct PointCloudData {
    pub points: Vec<Point>,
    // Maybe add reference images later ?
    pub camera_positions: Vec<Vec3>,
}

pub struct RenderState {
    camera: ViewerCamera,
    camera_positions: Vec<Vec3>,
    selected_camera: usize,

    points: Vec<Entity>,
    material_ref: ThreadSafeRef<Material>,
    mesh_ref: ThreadSafeRef<Mesh>,
}

impl BuildableApplicationState<PointCloudData> for RenderState {
    fn build(context: &mut morrigu::application::StateContext, data: PointCloudData) -> Self {
        let camera = Camera::builder().build(
            morrigu::components::camera::Projection::Perspective(PerspectiveData {
                horizontal_fov: f32::to_radians(64.5),
                near_plane: 0.001,
                far_plane: 1000.0,
            }),
            &Vec2::new(1280.0, 720.0),
        );
        let mut camera = ViewerCamera::new(camera);
        camera.set_focal_point(&Vec3::new(0.0, 0.0, 0.0));

        let shader_ref = Shader::from_path(
            Path::new("shaders/gen/gaussian.vert.spirv"),
            Path::new("shaders/gen/gaussian.frag.spirv"),
            &context.renderer.device,
        )
        .expect("Failed to create shader");
        let material_ref = Material::builder()
            .build::<Vertex>(
                &shader_ref,
                DescriptorResources::default(),
                context.renderer,
            )
            .expect("Failed to build material");

        let mesh_ref =
            Vertex::load_model_from_path_obj(Path::new("assets/sphere.obj"), context.renderer)
                .expect("Failed to load sphere model");

        let mut points = vec![];

        for point in data.points {
            let color_buffer = ThreadSafeRef::new(
                AllocatedBuffer::builder(std::mem::size_of::<Vec3>() as u64)
                    .build_with_data(point.color, context.renderer)
                    .expect("Failed to build color buffer"),
            );
            let sphere_rendering_ref = MeshRendering::new(
                &mesh_ref,
                &material_ref,
                DescriptorResources {
                    uniform_buffers: [
                        default_ubo_bindings(context.renderer).unwrap(),
                        (1, color_buffer),
                    ]
                    .into(),
                    ..Default::default()
                },
                context.renderer,
            )
            .expect("Failed to create mesh rendering");

            // let position = point.position * 10.0;
            let position = point.position * 2.0;
            let transform =
                Transform::from_trs(&position, &Quat::default(), &Vec3::new(0.005, 0.005, 0.005));

            let id = context
                .ecs_manager
                .world
                .spawn((transform, sphere_rendering_ref))
                .id();

            points.push(id);
        }

        Self {
            camera,
            camera_positions: data.camera_positions,
            selected_camera: 0,
            points,
            material_ref,
            mesh_ref,
        }
    }
}

impl ApplicationState for RenderState {
    fn on_attach(&mut self, context: &mut morrigu::application::StateContext) {
        context.ecs_manager.redefine_systems_schedule(|schedule| {
            schedule.add_systems(mesh_renderer::render_meshes::<Vertex>);
        });
    }

    fn on_update(
        &mut self,
        dt: std::time::Duration,
        context: &mut morrigu::application::StateContext,
    ) {
        self.camera.on_update(dt, context.window_input_state);
        context
            .ecs_manager
            .world
            .insert_resource(ResourceWrapper::new(context.window_input_state.clone()));
        context
            .ecs_manager
            .world
            .insert_resource(self.camera.mrg_camera);
    }

    fn on_update_egui(
        &mut self,
        _dt: std::time::Duration,
        context: &mut morrigu::application::EguiUpdateContext,
    ) {
        egui::Window::new("Settings and info").show(context.egui_context, |ui| {
            ui.label(format!("Number of points: {}", self.points.len()));

            egui::ComboBox::from_label("Select camera")
                .selected_text(format!("Camera #{}", self.selected_camera))
                .show_ui(ui, |ui| {
                    for (idx, _) in self.camera_positions.iter().enumerate() {
                        ui.selectable_value(
                            &mut self.selected_camera,
                            idx,
                            format!("Camera #{}", idx),
                        );
                    }
                });
            if ui.button("Snap to camera location").clicked() {
                let desired_pos = self.camera_positions.get(self.selected_camera).unwrap();
                // self.camera.distance = desired_pos.length();
                // self.camera.set_focal_point(&Vec3::new(0.0, 0.0, 0.0));
                // self.camera.mrg_camera.set_position(desired_pos);
                self.camera.mrg_camera.set_position(desired_pos);
                self.camera.lookat_temp(&Vec3::new(0.0, 0.0, 0.0));
                log::warn!(
                    "camera is now at {}, (stored is {})",
                    self.camera.mrg_camera.position(),
                    desired_pos
                );
            }
        });
    }

    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        _context: &mut morrigu::application::StateContext,
    ) {
        #[allow(clippy::single_match)] // Temporary
        match event {
            morrigu::application::Event::WindowEvent {
                event:
                    winit::event::WindowEvent::Resized(winit::dpi::PhysicalSize {
                        width, height, ..
                    }),
                ..
            } => {
                self.camera.on_resize(width, height);
            }
            _ => (),
        }
    }

    fn on_drop(&mut self, context: &mut morrigu::application::StateContext) {
        let mut query = context
            .ecs_manager
            .world
            .query::<&ThreadSafeRef<MeshRendering>>();
        for mrc in query.iter_mut(&mut context.ecs_manager.world) {
            mrc.lock().descriptor_resources.uniform_buffers[&0]
                .lock()
                .destroy(&context.renderer.device, &mut context.renderer.allocator());
            mrc.lock().descriptor_resources.uniform_buffers[&1]
                .lock()
                .destroy(&context.renderer.device, &mut context.renderer.allocator());
            mrc.lock().destroy(context.renderer)
        }

        self.mesh_ref.lock().destroy(context.renderer);

        self.material_ref
            .lock()
            .shader_ref
            .lock()
            .destroy(&context.renderer.device);
        self.material_ref.lock().destroy(context.renderer);
    }
}
