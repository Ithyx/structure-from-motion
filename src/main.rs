use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use morrigu::application::ApplicationBuilder;
use opencv as cv;

use crate::{
    pose::extract_pose,
    render_state::{PointCloudData, RenderState},
    sfm::generate_point_cloud,
};

mod camera;
mod pose;
mod render_state;
mod sfm;

pub type Image = cv::core::Mat;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLI {
    /// The path to the folder containing the images you would like to use.
    #[arg(short, long, value_name = "FOLDER")]
    pub data_path: PathBuf,
    /// The name (NOT PATH) of the file containing pose data in the folder specified. Needs to follow the
    /// templering dataset pose format. Defaults to "pose.txt"
    #[arg(short, long)]
    pub pose_file: Option<String>,
}

fn init_logging() {
    #[cfg(debug_assertions)]
    let log_level = ("trace", flexi_logger::Duplicate::Debug);
    #[cfg(not(debug_assertions))]
    let log_level = ("info", flexi_logger::Duplicate::Info);

    let file_spec = flexi_logger::FileSpec::default().suppress_timestamp();

    let _logger = flexi_logger::Logger::try_with_env_or_str(log_level.0)
        .expect("Failed to setup logging")
        .log_to_file(file_spec)
        .write_mode(flexi_logger::WriteMode::BufferAndFlush)
        .duplicate_to_stdout(log_level.1)
        .set_palette("b9;3;2;8;7".to_owned())
        .start()
        .expect("Failed to build logger");
}

fn main() {
    init_logging();

    let cli = CLI::parse();

    let mut file_paths: Vec<_> = std::fs::read_dir(&cli.data_path)
        .expect("Failed to read files in the specified folder")
        .flatten()
        .collect();
    file_paths.sort_by_key(|a| a.path());

    let pose_file = file_paths
        .iter()
        .find(|entry| {
            &entry.file_name().to_string_lossy().to_string()
                == cli.pose_file.as_ref().unwrap_or(&"pose.txt".to_owned())
        })
        .expect("Failed to find pose file")
        .path();
    log::info!("Found pose file {}", pose_file.to_string_lossy());

    let (poses, camera_positions) =
        extract_pose(pose_file).expect("Failed to read pose information");

    log::info!("loading images from: {}", cli.data_path.to_string_lossy());
    let images: Vec<Image> = file_paths
        .into_iter()
        .flat_map(|entry| -> Result<Image> {
            match &*entry
                .path()
                .extension()
                .context("Failed to parse file extension")?
                .to_string_lossy()
            {
                "png" | "jpg" | "jpeg" => {
                    let img = cv::imgcodecs::imread(
                        &entry.path().to_string_lossy(),
                        cv::imgcodecs::IMREAD_COLOR,
                    )?;
                    // let mut rotated_img = cv::core::Mat::default();
                    // cv::core::rotate(
                    //     &img,
                    //     &mut rotated_img,
                    //     cv::core::RotateFlags::ROTATE_90_COUNTERCLOCKWISE.into(),
                    // )
                    // .expect("Failed to rotate image");
                    log::debug!("\tloaded {}", entry.path().to_string_lossy());
                    // Ok(rotated_img)
                    Ok(img)
                }

                _ => None.context("Invalid file extention")?,
            }
        })
        .collect();
    log::info!("loaded {} images", images.len());

    let points = generate_point_cloud(images, poses).expect("Failed to generate cloud point");

    ApplicationBuilder::new()
        .with_window_name("Point cloud viewer")
        .with_dimensions(1280, 720)
        .with_application_name("IFT6142 project")
        .with_application_version(0, 1, 0)
        .build_and_run_inplace::<RenderState, PointCloudData>(PointCloudData {
            points,
            camera_positions,
        });
}
