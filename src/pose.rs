use std::path::PathBuf;

use anyhow::Result;
use cv::prelude::MatExprTraitConst;
use morrigu::math_types::Vec3;
use opencv as cv;

pub fn extract_pose(pose_file_path: PathBuf) -> Result<(Vec<cv::core::Mat>, Vec<Vec3>)> {
    let file_contents = std::fs::read_to_string(pose_file_path)?;

    let mut lines = file_contents.lines();
    lines.next(); // ignore first line

    log::info!("Extracting pose data for images");
    let mut poses = vec![];
    let mut camera_positions = vec![];
    for line in lines {
        let mut params = line.split(' ');
        let filename = params.next().unwrap(); // ignore filename

        let k_vals = params
            .clone()
            .take(9)
            .map(|s| {
                s.parse::<f32>()
                    .expect("Failed to parse float value for the K matrix")
            })
            .collect::<Vec<_>>();
        let k = cv::core::Mat::from_slice_rows_cols(&k_vals, 3, 3)?;
        let params = params.skip(9);

        let mut rt_vals = params
            .clone()
            .take(9)
            .map(|s| {
                s.parse::<f32>()
                    .expect("Failed to parse float value for the R matrix")
            })
            .collect::<Vec<_>>();
        let mut params = params.skip(9);

        let position = Vec3::new(
            params.next().unwrap().parse::<f32>().unwrap(),
            params.next().unwrap().parse::<f32>().unwrap(),
            params.next().unwrap().parse::<f32>().unwrap(),
        );

        camera_positions.push(position);
        rt_vals.insert(3, position.x);
        rt_vals.insert(7, position.y);
        rt_vals.insert(11, position.z);

        let rt = cv::core::Mat::from_slice_rows_cols(&rt_vals, 3, 4)?;

        let pose = (k * rt).into_result()?.to_mat()?;

        log::debug!("\tExtracted values for {}:", filename);
        log::debug!("\t\tk: {:?}", k_vals);
        log::debug!("\t\trt: {:?}", rt_vals);
        log::debug!(
            "\t\t -> {:?}",
            pose.iter::<f32>()?.map(|(_, val)| val).collect::<Vec<_>>()
        );

        poses.push(pose);
    }
    log::info!("Extracted pose data for images");

    Ok((poses, camera_positions))
}
