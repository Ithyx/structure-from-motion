# Structure from Motion: a re-implementation

## What
This repo contains a re-implementation of the [*Structure from Motion*](https://en.wikipedia.org/wiki/Structure_from_motion) algoritm, which attempts to rebuild an object using views of this object taken from different viewpoints.
This was made as a final project in the computer vision course [IFT 6145](https://diro.umontreal.ca/programmes-cours/cours-horaires/details-de-certains-cours/vision-tridimensionnelle/).

## How
This project uses rust, with bindings with the OpenCV library which does the heavy lifting, and uses 
- SIFT descriptors
- FLANN based matching to establish correspondances
- 2-view triangulation to finish re-building the coordinates
- [morrigu-rs](https://github.com/TableauBits/morrigu-rs) to view the point-cloud

## Build and run
First and foremost, you will need a version of OpenCV compiled with the SFM contib module. This was tested with OpenCV 4.8.1 (with matching opencv_contrib version). If you attempt to run this code, make sure the environment variables in the file `.cargo/config.toml` point to the needed directories (it should work out of the box if you build OpenCV manually and install it to the default location (on linux)) If you built openCV manually, make sure ld can see where the needed libraries are.
After that, you will need the vulkan SDK (and the validation layers if running in debug mode), as well as glslc on your path for morrigu-rs to work properly.

As for the dataset, the program is heavily fitted for the [*TempleRing dataset*](https://vision.middlebury.edu/mview/data/), which is included in `data`, making assumptions, such as the pose format and view order. With that said, it is definitely possible to remove these assumptions if needed.
An example invocation would look like:
```bash
cargo run --release -- -d data/templeRing/
```
