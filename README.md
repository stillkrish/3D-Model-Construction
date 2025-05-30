# 3D Model Reconstruction Using 10-bit Non-inverse Binary Structured Light

A computer vision project exploring 3D surface reconstruction of an object using structured light patterns and stereo camera configuration.
Our target object was an Airpods Max Case

## Project Overview

This project implements 3D reconstruction using 10-bit non-inverse binary structured light patterns with two calibrated cameras. We successfully generated point clouds and polygonal meshes, though the final models exhibited noticeable errors due to suboptimal image quality and challenging binary code thresholding conditions.

## Setup

### Hardware Configuration
- **Stereo camera setup**: Two cameras in stereo configuration
- **Projection system**: Digital projector positioned between cameras
- **Calibration target**: 7x9 chessboard pattern (30mm squares)
- **Background**: Dark blue non-reflective backdrop to reduce ambient light
- **Target object**: AirPods Max case

### Image Capture Process
1. **Calibration**: 20 chessboard images (10 from each camera angle)
2. **Structured light capture**: 5 rotations × 20 images per rotation
   - 10 horizontal stripe patterns
   - 10 vertical stripe patterns
3. **Reference images**: Regular lighting shots with and without object

## Technical Approach

### 1. Camera Calibration
- **Intrinsic parameters**: Focal length, principal point, skew, distortion coefficients
- **Extrinsic parameters**: Camera pose estimation using decoded binary patterns
- Standard checkerboard calibration techniques

### 2. Pattern Decoding
- Binary code pattern decoding to generate unique pixel indices
- Creation of three outputs:
  - **Code**: Decoded binary patterns representing 3D position
  - **Mask**: Decoding confidence/quality indicator  
  - **Color mask**: Texture and material property preservation

### 3. 3D Reconstruction Pipeline
1. **Point correspondence**: Match pixels between stereo camera views
2. **Triangulation**: Convert 2D matched points to 3D coordinates
3. **Point cloud generation**: Create sparse 3D representation
4. **Mesh creation**: Connect points to form triangular surface mesh
5. **Filtering**: Remove outlier triangles and noise

## Technologies Used

- **Languages**: Python, MATLAB
- **Libraries**: OpenCV, NumPy
- **Visualization**: Sketchfab.com for 3D model rendering
- **Output format**: PLY files

## Lessons Learned

### Key Issues
1. **Pattern choice**: 10-bit non-inverse binary patterns proved suboptimal
2. **Thresholding sensitivity**: Without inverse images, binary decoding became unreliable
3. **Image quality**: Inconsistent lighting and exposure affected pattern clarity

### Recommended Improvements
1. **Gray Code patterns**: More robust decoding with single-bit differences between consecutive patterns
2. **Controlled lighting**: Darker environment with projector-only illumination
3. **Hardware upgrades**: Brighter projector, higher-resolution cameras, better lenses
4. **Exposure optimization**: Careful camera settings for clearer on/off state distinction

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Camera calibration tools
- Digital projector
- Stereo camera setup

## Future Work

- Implement Gray-coded structured light patterns
- Improve environmental controls and lighting setup
- Experiment with additional pattern sequences
- Optimize camera exposure and focus settings
- Explore alternative reconstruction algorithms

---

*This project demonstrates fundamental stereo structured light scanning principles while highlighting the importance of pattern selection and environmental control in 3D reconstruction systems.*
