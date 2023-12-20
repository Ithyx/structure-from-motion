use crate::{render_state, Image};
use anyhow::Result;

use cv::core::{DMatch, Point2f, Vec3b, Vector};
#[allow(unused)]
use itertools::Itertools;
use morrigu::math_types::Vec3;
use opencv as cv;
use opencv::prelude::*;

fn extract_features(
    images: &[Image],
) -> Result<(
    Vec<cv::core::Vector<cv::core::KeyPoint>>,
    Vec<cv::core::Mat>,
)> {
    let mut sift_detector = cv::features2d::SiftFeatureDetector::create_def()?;

    let mut keypoints = Vec::<_>::with_capacity(images.len());
    let mut descriptors = Vec::<_>::with_capacity(images.len());

    log::info!("Finding keypoints in images");
    for (idx, image) in images.iter().enumerate() {
        let mut img_keypoints = cv::core::Vector::<cv::core::KeyPoint>::new();
        let mut img_descriptors = cv::core::Mat::default();
        sift_detector.detect_and_compute_def(
            &image,
            &cv::core::no_array(),
            &mut img_keypoints,
            &mut img_descriptors,
        )?;

        log::debug!(
            "\tFound {} keypoints in image #{}",
            img_keypoints.len(),
            idx + 1
        );

        keypoints.push(img_keypoints);
        descriptors.push(img_descriptors);
    }
    log::info!("Computed keypoints in all images");

    Ok((keypoints, descriptors))
}

fn find_matches(
    descriptors1: &cv::core::Mat,
    descriptors2: &cv::core::Mat,
) -> Result<cv::core::Vector<cv::core::DMatch>> {
    let index_params = cv::flann::KDTreeIndexParams::new(5)?;
    let search_params = cv::flann::SearchParams::new_def()?;
    let matcher = cv::features2d::FlannBasedMatcher::new(
        &cv::core::Ptr::new(index_params.into()),
        &cv::core::Ptr::new(search_params),
    )?;

    let mut matches = cv::core::Vector::<cv::core::Vector<cv::core::DMatch>>::new();
    matcher.knn_train_match_def(descriptors1, descriptors2, &mut matches, 2)?;

    let mut matches: Vec<cv::core::DMatch> = matches
        .iter()
        .flat_map(|img_match| {
            match img_match.get(0).unwrap().distance < 0.7 * img_match.get(1).unwrap().distance {
                // match true {
                true => Some(img_match.get(0).unwrap()),
                false => None,
            }
        })
        .collect();

    matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    let matches: Vector<DMatch> = matches.into_iter().take(100).collect();

    log::debug!("\tfound {} matches", matches.len());
    Ok(matches)
}

fn format_point(
    cv_point: &cv::core::Mat,
    image1: &cv::core::Mat,
    image2: &cv::core::Mat,
    keypoint1: &cv::core::Point2f,
    keypoint2: &cv::core::Point2f,
) -> Result<render_state::Point> {
    let position = Vec3::new(
        *cv_point.at::<f64>(0).unwrap() as f32,
        *cv_point.at::<f64>(1).unwrap() as f32,
        *cv_point.at::<f64>(2).unwrap() as f32,
    );

    let color1 = image1.at_2d::<Vec3b>(keypoint1.y.floor() as i32, keypoint1.x.floor() as i32)?;
    let color2 = image2.at_2d::<Vec3b>(keypoint2.y.floor() as i32, keypoint2.x.floor() as i32)?;
    let color = Vec3::new(
        (*color1.last().unwrap() as f32 + *color2.last().unwrap() as f32) / (2.0 * u8::MAX as f32),
        (*color1.get(1).unwrap() as f32 + *color2.get(1).unwrap() as f32) / (2.0 * u8::MAX as f32),
        (*color1.first().unwrap() as f32 + *color2.first().unwrap() as f32)
            / (2.0 * u8::MAX as f32),
    );

    Ok(render_state::Point { position, color })
}

fn triangulate_points(
    image1: &cv::core::Mat,
    image2: &cv::core::Mat,
    pose1: cv::core::Mat,
    pose2: cv::core::Mat,
    keypoints1: &cv::core::Vector<cv::core::KeyPoint>,
    keypoints2: &cv::core::Vector<cv::core::KeyPoint>,
    matches: &cv::core::Vector<cv::core::DMatch>,
) -> Result<Vec<render_state::Point>> {
    let mut formatted_points = Vec::with_capacity(matches.len());

    let mut point_3d = cv::core::Mat::default();
    let poses: cv::core::Vector<cv::core::Mat> = vec![pose1, pose2].into();
    let mut points_2d = Vector::<Mat>::new();
    points_2d.push(Mat::default());
    points_2d.push(Mat::default());
    for img_match in matches {
        let left_keypoint = keypoints1
            .get(img_match.query_idx.try_into().unwrap())
            .unwrap()
            .pt();
        let slice = [left_keypoint.x, left_keypoint.y];
        points_2d.set(0, Mat::from_slice_rows_cols(&slice, 2, 1)?)?;

        let right_keypoint = keypoints2
            .get(img_match.train_idx.try_into().unwrap())
            .unwrap()
            .pt();
        let slice = [right_keypoint.x, right_keypoint.y];
        points_2d.set(1, Mat::from_slice_rows_cols(&slice, 2, 1)?)?;

        cv::sfm::triangulate_points(&points_2d, &poses, &mut point_3d)?;
        formatted_points.push(format_point(
            &point_3d,
            image1,
            image2,
            &left_keypoint,
            &right_keypoint,
        )?);
    }

    Ok(formatted_points)
}

#[allow(dead_code)]
pub fn hardcode_triangulation(
    pose1: cv::core::Mat,
    pose2: cv::core::Mat,
) -> Result<Vec<render_state::Point>> {
    let left_points: [[f32; 2]; 10] = [
        [132.0, 112.0],
        [431.0, 157.0],
        [510.0, 197.0],
        [151.0, 378.0],
        [219.0, 200.0],
        [129.0, 232.0],
        [140.0, 322.0],
        [406.0, 373.0],
        [405.0, 285.0],
        [427.0, 173.0],
    ];
    let right_points: [[f32; 2]; 10] = [
        [144.0, 96.0],
        [433.0, 128.0],
        [513.0, 169.0],
        [137.0, 374.0],
        [219.0, 216.0],
        [129.0, 202.0],
        [136.0, 302.0],
        [404.0, 380.0],
        [404.0, 284.0],
        [430.0, 144.0],
    ];

    let poses: cv::core::Vector<cv::core::Mat> = vec![pose1, pose2].into();
    let mut cv_point = cv::core::Mat::default();
    let mut points_2d = Vector::<Mat>::new();
    let mut formatted_points = Vec::with_capacity(left_points.len());

    points_2d.push(Mat::default());
    points_2d.push(Mat::default());
    for match_ in left_points.iter().zip(right_points.iter()) {
        points_2d.set(0, Mat::from_slice_rows_cols(match_.0, 2, 1)?)?;
        points_2d.set(1, Mat::from_slice_rows_cols(match_.1, 2, 1)?)?;

        cv::sfm::triangulate_points(&points_2d, &poses, &mut cv_point)?;

        let position = Vec3::new(
            *cv_point.at::<f64>(0).unwrap() as f32,
            *cv_point.at::<f64>(1).unwrap() as f32,
            *cv_point.at::<f64>(2).unwrap() as f32,
        );
        formatted_points.push(render_state::Point {
            position,
            color: Vec3::new(0.8, 0.2, 0.2),
        });
    }

    Ok(formatted_points)
}

#[allow(dead_code)]
pub fn test_triangulation(
    camera1: Mat,
    camera2: Mat,
    point2d1: Point2f,
    point2d2: Point2f,
    expected3d: Vec3,
) -> Result<()> {
    log::info!("left match: {:?}", point2d1);
    log::info!("right match: {:?}", point2d2);

    let points_2d_values_left = [point2d1.x, point2d1.y];
    let points_2d_left = Mat::from_slice_rows_cols(&points_2d_values_left, 2, 1)?;
    let points_2d_values_right = [point2d2.x, point2d2.y];
    let points_2d_right = Mat::from_slice_rows_cols(&points_2d_values_right, 2, 1)?;
    let points_2d: Vector<Mat> = vec![points_2d_left, points_2d_right].into();

    log::info!(
        "camera1 ({:?}): {:?}",
        camera1.size()?,
        camera1
            .iter::<f32>()?
            .map(|(_, value)| value)
            .collect::<Vec<_>>()
    );
    log::info!(
        "camera2: {:?}",
        camera2
            .iter::<f32>()?
            .map(|(_, value)| value)
            .collect::<Vec<_>>()
    );
    let poses: Vector<Mat> = vec![camera1, camera2].into();

    let mut point_3d = cv::core::Mat::default();
    cv::sfm::triangulate_points(&points_2d, &poses, &mut point_3d)?;

    let actual_3d = Vec3::new(
        *point_3d.at::<f64>(0)? as f32,
        *point_3d.at::<f64>(1)? as f32,
        *point_3d.at::<f64>(2)? as f32,
    );

    log::info!("Expected {}, got {}", expected3d, actual_3d);

    Ok(())
}

pub fn generate_point_cloud(
    images: Vec<Image>,
    poses: Vec<cv::core::Mat>,
) -> Result<Vec<render_state::Point>> {
    let (keypoints, descriptors) = extract_features(&images)?;

    let output_subfolder = "out";
    let should_output_images = std::path::Path::new(output_subfolder).exists()
        || match std::fs::create_dir(output_subfolder) {
            Ok(_) => true,
            Err(_) => {
                log::error!("Failed to create output directory, no images will be generated");
                false
            }
        };

    let mut points = vec![];
    log::info!("Generating points");
    // for index_pair in (0..images.len()).combinations(2) {
    for index_pair in (0..images.len()).collect::<Vec<_>>().windows(2) {
        let left_idx = index_pair[0];
        let right_idx = index_pair[1];
        log::debug!("\tmatching between {} and {}", left_idx, right_idx);
        let matches = find_matches(&descriptors[left_idx], &descriptors[right_idx])?;

        if should_output_images {
            let mut output_image = cv::core::Mat::default();
            cv::features2d::draw_matches_def(
                &images[left_idx],
                &keypoints[left_idx],
                &images[right_idx],
                &keypoints[right_idx],
                &matches,
                &mut output_image,
            )?;
            cv::imgcodecs::imwrite_def(
                &format!("./{}/{}-{}.png", output_subfolder, left_idx, right_idx),
                &output_image,
            )?;

            // let test_match = matches.get(7)?;
            // cv::features2d::draw_matches_def(
            //     &images[left_idx],
            //     &keypoints[left_idx],
            //     &images[right_idx],
            //     &keypoints[right_idx],
            //     &vec![test_match].into(),
            //     &mut output_image,
            // )?;
            // cv::imgcodecs::imwrite_def(
            //     &format!("./{}/{}-{}_TEST.png", output_subfolder, left_idx, right_idx),
            //     &output_image,
            // )?;
        }

        // test_triangulation(
        //     poses[left_idx].clone(),
        //     poses[right_idx].clone(),
        //     keypoints[left_idx].get(test_match.query_idx as usize)?.pt(),
        //     keypoints[right_idx]
        //         .get(test_match.train_idx as usize)?
        //         .pt(),
        //     Vec3::new(0.0, 0.0, 0.0),
        // )?;

        points.append(&mut triangulate_points(
            &images[left_idx],
            &images[right_idx],
            poses[left_idx].clone(),
            poses[right_idx].clone(),
            &keypoints[left_idx],
            &keypoints[right_idx],
            &matches,
        )?);

        // points.append(&mut hardcode_triangulation(
        //     poses[left_idx].clone(),
        //     poses[right_idx].clone(),
        // )?);
    }
    log::info!("Generated {} points", points.len());

    Ok(points)
}
