# 4D Volumetric Retina Simulation

This project presents a novel simulation and visualization of perception in a four-dimensional (4D) spatial environment, focusing on how a hypothetical 4D creature might "see" its world through a volumetric 3D retina.

## Motivation

Current visualizations of 4D space often present wireframe models of 4D objects (like hypercubes/tesseracts) or depict simple 3D cross-sections (hyperplane slices) of these objects. While a few explorations have ventured into ray-tracing 4D scenes, a critical aspect often overlooked is the nature of the sensory organ itself. Analogous to how 3D beings perceive a 3D world via 2D retinas, it is posited that a 4D creature would possess a 3D (volumetric) retina.

This simulation aims to bridge this gap by:
1.  Implementing a physically-based 4D path tracing engine to model light interaction within a 4D scene.
2.  Simulating image formation onto a defined 3D volumetric retina.
3.  Introducing a model for information acuity across this 3D retina, assuming that sensitivity might be highest at its center and fall off towards its periphery, modeled here with a Gaussian distribution.
4.  Combining multiple 2D slices from this simulated 3D retina, weighted by the acuity model, into a single composite 2D image that offers a richer, more nuanced representation of 4D visual perception than previously available.

As far as I know, this approach provides one of the most detailed and conceptually faithful attempts to visualize what a 4D being might perceive.

## Simulation Details

*   **Core Engine:** 4D Path Tracer implemented in Python using Taichi Lang for GPU acceleration.
*   **Scene:** Features a rotating 4D hypercube (tesseract) and a 4D plane, illuminated by a 4D point light source.
*   **4D Eye Model:**
    *   A 4D camera casts rays from a point in 4D space.
    *   These rays pass through points defining a 3D volumetric "retina" located in a specific 3D hyperplane (e.g., W=0).
*   **Retinal Image Formation:**
    *   Each "voxel" (or point) on the 3D retina collects light information gathered by a 4D ray passing through it, determined by the path tracing simulation (including direct lighting, shadows, and multiple light bounces).
*   **Volumetric Acuity Model:**
    *   The contribution of information from different parts of the 3D retina to the final perceived image is weighted. This simulation assumes a Gaussian fall-off in information prominence from the center of the 3D retina towards its edges. This models the idea that, similar to foveal vision in humans, the "center" of the 4D creature's 3D retinal "gaze" would be most acute.
*   **Final Image Compositing:**
    *   Multiple 2D slices are taken from the simulated 3D retinal volume.
    *   These 2D slices are additively blended, with each slice's contribution weighted according to the Gaussian acuity model applied to its position within the 3D retina. The result is a single 2D image that attempts to convey the volumetric nature of the 4D perception.
    
## Resulting Animation

[![4D Volumetric Retina Simulation](preview.gif)](https://github.com/volotat/4DRender/raw/main/result.mp4)

*Click on the image above to download and view the full video.*

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{Borsky_4D_Volumetric_Retina_2025,
  author = {Borsky, Alexey},
  month = {5},
  title = {{4D Volumetric Retina Simulation}},
  url = {https://github.com/volotat/4DRender},
  version = {1.0.0},
  year = {2025}
}
```
