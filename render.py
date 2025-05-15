import taichi as ti
import taichi.math as tm
import math
import time
import os
import numpy as np 
import cv2        
from scipy.stats import norm 

# --- Configuration for Video Rendering ---
VIDEO_TOTAL_FRAMES = 757 # Full rotation of the hypercube
VIDEO_FPS = 24 # Desired FPS of the output video
ACCUMULATION_SECONDS_PER_VIDEO_FRAME = 2.0 # How long to accumulate for each video frame

TEMP_FRAME_DIR = "temp_video_frames"
SAVE_UPSCALED_FRAMES = True # True to save upscaled, False to save RENDER_RESOLUTION frames

def generate_retina_slices(num_slices, min_offset=-0.2, max_offset=0.2):
    """
    Generate retina z-slice offsets and corresponding weights with a normal distribution
    
    Args:
        num_slices: Number of z-slices to generate
        min_offset: Minimum z offset value (default: -0.2)
        max_offset: Maximum z offset value (default: 0.2)
        
    Returns:
        tuple: (offset_list, weight_list)
    """
    # Generate evenly spaced points between min and max
    offset_list = np.linspace(min_offset, max_offset, num_slices)
    
    # Calculate weights using normal distribution (centered at 0)
    # Standard deviation set to make the distribution cover the range appropriately
    std_dev = (max_offset - min_offset) / 6  # ~99.7% of values within 3 standard deviations
    
    # Calculate the probability density at each offset point
    weight_list = norm.pdf(offset_list, loc=0, scale=std_dev)
    
    # Normalize weights for better control (maximum weight = 1.0)
    weight_list = weight_list / np.max(weight_list)
    
    return offset_list.tolist(), weight_list.tolist()

# Initialize Taichi
try:
    ti.init(arch=ti.gpu, device_memory_GB=2) # Increased memory for longer runs if needed
    print("Taichi running on GPU")
except Exception as e:
    print(f"GPU initialization failed: {e}. Falling back to CPU.")
    ti.init(arch=ti.cpu)
    print("Taichi running on CPU")

# --- Screen and Rendering Resolution ---
ASPECT_RATIO = 9 / 16 # Portrait aspect ratio for the video
RENDER_HEIGHT = 240
RENDER_WIDTH = int(ASPECT_RATIO * RENDER_HEIGHT)

# --- GUI Window Size (Larger) ---
GUI_SCALE_FACTOR = 3 
GUI_HEIGHT = RENDER_HEIGHT * GUI_SCALE_FACTOR
GUI_WIDTH = RENDER_WIDTH * GUI_SCALE_FACTOR

_WIDTH = RENDER_WIDTH
_HEIGHT = RENDER_HEIGHT

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(_WIDTH, _HEIGHT)) 
accumulated_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(_WIDTH, _HEIGHT))
averaged_low_res_image = ti.Vector.field(3, dtype=ti.f32, shape=(_WIDTH, _HEIGHT))
if SAVE_UPSCALED_FRAMES:
    frame_save_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(GUI_WIDTH, GUI_HEIGHT))
else:
    frame_save_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(_WIDTH, _HEIGHT))


# --- CAMERA AND SCENE SETTINGS ---
CAMERA_EYE_W_COORD = -3.0
HYPERCUBE_CENTER_CONST = tm.vec4(0.0, 0.0, 0.0, 0.0) 
HYPERCUBE_SIZE_CONST = 0.45
LIGHT_POS_CONST = tm.vec4(1.5, 1.5, -1.5, -1.0)
LIGHT_COLOR_CONST = tm.vec3(5.0, 5.0, 4.0) 
AMBIENT_STRENGTH_CONST = 0.05 
OBJECT_COLOR_CONST = tm.vec3(0.95, 0.75, 0.75) 
BACKGROUND_COLOR_CONST = tm.vec3(0.0, 0.0, 0.0) 
EPS = 1e-4; INF = 1e9; PI = math.pi 
PLANE_W_COORD_CONST = 0.25
PLANE_NORMAL_CONST = tm.vec4(0.0, 0.0, 0.0, -1.0)
PLANE_COLOR_CONST = tm.vec3(0.75, 0.75, 0.75) 
MAX_BOUNCES = 3

# --- 4D Rotation Matrix Helper Functions ---
@ti.func
def create_rotation_xw(angle: ti.f32) -> tm.mat4:
    c, s = tm.cos(angle), tm.sin(angle)
    return tm.mat4([[c,0,0,-s],[0,1,0,0],[0,0,1,0],[s,0,0,c]])
@ti.func
def create_rotation_yz(angle: ti.f32) -> tm.mat4:
    c, s = tm.cos(angle), tm.sin(angle)
    return tm.mat4([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
@ti.func
def create_rotation_xy(angle: ti.f32) -> tm.mat4:
    c, s = tm.cos(angle), tm.sin(angle)
    return tm.mat4([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
@ti.func
def create_rotation_zw(angle: ti.f32) -> tm.mat4:
    c, s = tm.cos(angle), tm.sin(angle)
    return tm.mat4([[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]])

# --- Ray-Object Intersection Functions ---
@ti.func
def intersect_hypercube( 
    ray_origin_world: tm.vec4, ray_dir_world: tm.vec4,
    obj_center_world: tm.vec4, obj_size: ti.f32,
    object_rotation_matrix: tm.mat4, inv_object_rotation_matrix: tm.mat4
): 
    ray_origin_obj = inv_object_rotation_matrix @ (ray_origin_world - obj_center_world)
    ray_dir_obj = inv_object_rotation_matrix @ ray_dir_world
    min_bounds = tm.vec4(-obj_size * 0.5)
    max_bounds = tm.vec4(obj_size * 0.5)
    inv_dir = 1.0 / ray_dir_obj
    t_bottom = (min_bounds - ray_origin_obj) * inv_dir
    t_top = (max_bounds - ray_origin_obj) * inv_dir
    t_min_axis = tm.min(t_bottom, t_top)
    t_max_axis = tm.max(t_bottom, t_top)
    t_enter = max(t_min_axis.x, t_min_axis.y, t_min_axis.z, t_min_axis.w)
    t_exit = min(t_max_axis.x, t_max_axis.y, t_max_axis.z, t_max_axis.w)
    hit_t = -1.0
    normal_obj = tm.vec4(0.0)
    if t_enter < t_exit and t_exit > EPS:
        if t_enter > EPS: hit_t = t_enter
        elif t_exit > EPS: hit_t = t_exit
    if hit_t > 0.0:
        hit_point_obj = ray_origin_obj + hit_t * ray_dir_obj
        abs_dist_to_surface = abs(abs(hit_point_obj) - (obj_size * 0.5))
        min_abs_dist = min(abs_dist_to_surface.x, abs_dist_to_surface.y, abs_dist_to_surface.z, abs_dist_to_surface.w)
        comparison_eps = EPS * 0.01
        if abs(abs_dist_to_surface.x - min_abs_dist) < comparison_eps: normal_obj.x = tm.sign(hit_point_obj.x)
        elif abs(abs_dist_to_surface.y - min_abs_dist) < comparison_eps: normal_obj.y = tm.sign(hit_point_obj.y)
        elif abs(abs_dist_to_surface.z - min_abs_dist) < comparison_eps: normal_obj.z = tm.sign(hit_point_obj.z)
        else: normal_obj.w = tm.sign(hit_point_obj.w)
        if normal_obj.norm_sqr() < EPS*EPS: 
            abs_coords = abs(hit_point_obj)
            if abs_coords.x >= abs_coords.y and abs_coords.x >= abs_coords.z and abs_coords.x >= abs_coords.w: normal_obj.x = tm.sign(hit_point_obj.x)
            elif abs_coords.y >= abs_coords.z and abs_coords.y >= abs_coords.w: normal_obj.y = tm.sign(hit_point_obj.y)
            elif abs_coords.z >= abs_coords.w: normal_obj.z = tm.sign(hit_point_obj.z)
            else: normal_obj.w = tm.sign(hit_point_obj.w)
            if normal_obj.norm_sqr() < EPS*EPS and obj_size > 0:
                 normal_obj.x = 1.0
    return hit_t, normal_obj

@ti.func
def intersect_plane(
    ray_origin_world: tm.vec4, ray_dir_world: tm.vec4,
    plane_w: ti.f32 
) -> ti.f32:
    t_hit = -1.0
    denominator = ray_dir_world.w
    if abs(denominator) > EPS:
        t = (plane_w - ray_origin_world.w) / denominator
        if t > EPS: t_hit = t
    return t_hit

@ti.func
def sample_cosine_weighted_hemisphere_direction(normal: tm.vec4) -> tm.vec4:
    result_dir = normal 
    s_on_3_sphere = tm.vec4(0.0)
    len_sq = 0.0
    max_rejection_iters = 10 
    found_s_on_3_sphere = False
    for _ in range(max_rejection_iters):
        v1 = ti.random(ti.f32) * 2.0 - 1.0; v2 = ti.random(ti.f32) * 2.0 - 1.0
        v3 = ti.random(ti.f32) * 2.0 - 1.0; v4 = ti.random(ti.f32) * 2.0 - 1.0
        len_sq = v1*v1 + v2*v2 + v3*v3 + v4*v4
        if len_sq <= 1.0 and len_sq > EPS*EPS :
            inv_len = 1.0 / tm.sqrt(len_sq)
            s_on_3_sphere = tm.vec4(v1*inv_len, v2*inv_len, v3*inv_len, v4*inv_len)
            found_s_on_3_sphere = True
            break 
    if found_s_on_3_sphere:
        candidate_dir = normal + s_on_3_sphere
        if candidate_dir.norm_sqr() > EPS*EPS: 
            bounced_dir_candidate = tm.normalize(candidate_dir)
            if tm.dot(bounced_dir_candidate, normal) > EPS : 
                result_dir = bounced_dir_candidate
    return result_dir

@ti.func
def trace_path(
    initial_ray_origin: tm.vec4, initial_ray_dir: tm.vec4,
    hypercube_rotation_matrix: tm.mat4,
    inv_hypercube_rotation_matrix: tm.mat4
) -> tm.vec3:
    accumulated_color = tm.vec3(0.0)
    path_throughput = tm.vec3(1.0) 
    current_ray_origin = initial_ray_origin
    current_ray_dir = initial_ray_dir
    for bounce_idx in range(MAX_BOUNCES):
        closest_t = INF; hit_type = 0 
        surface_normal_world = tm.vec4(0.0); surface_albedo = tm.vec3(0.0)
        normal_cube_obj_if_hit = tm.vec4(0.0) 
        t_cube, normal_cube_obj = intersect_hypercube(
            current_ray_origin, current_ray_dir, HYPERCUBE_CENTER_CONST, HYPERCUBE_SIZE_CONST,
            hypercube_rotation_matrix, inv_hypercube_rotation_matrix)
        if t_cube > EPS and t_cube < closest_t:
            closest_t = t_cube; hit_type = 1
            normal_cube_obj_if_hit = normal_cube_obj 
            surface_albedo = OBJECT_COLOR_CONST
        t_plane = intersect_plane(current_ray_origin, current_ray_dir, PLANE_W_COORD_CONST)
        if t_plane > EPS and t_plane < closest_t: 
            closest_t = t_plane; hit_type = 2
            surface_albedo = PLANE_COLOR_CONST
        if hit_type == 1: surface_normal_world = tm.normalize(hypercube_rotation_matrix @ normal_cube_obj_if_hit)
        elif hit_type == 2: surface_normal_world = PLANE_NORMAL_CONST
        if hit_type == 0: 
            if bounce_idx == 0: accumulated_color += path_throughput * BACKGROUND_COLOR_CONST
            break 
        hit_point_world = current_ray_origin + closest_t * current_ray_dir
        if bounce_idx == 0: accumulated_color += path_throughput * AMBIENT_STRENGTH_CONST * surface_albedo
        if surface_normal_world.norm_sqr() > EPS*EPS: 
            light_vec_world = tm.normalize(LIGHT_POS_CONST - hit_point_world)
            dist_to_light = (LIGHT_POS_CONST - hit_point_world).norm()
            cos_theta_direct = tm.dot(surface_normal_world, light_vec_world)
            if cos_theta_direct > EPS: 
                shadow_ray_origin = hit_point_world + surface_normal_world * EPS * 10.0
                in_shadow = False
                t_shadow_cube, _ = intersect_hypercube(
                    shadow_ray_origin, light_vec_world, HYPERCUBE_CENTER_CONST, HYPERCUBE_SIZE_CONST,
                    hypercube_rotation_matrix, inv_hypercube_rotation_matrix)
                if t_shadow_cube > EPS and t_shadow_cube < dist_to_light: in_shadow = True
                if not in_shadow:
                    direct_light_val = (LIGHT_COLOR_CONST / PI) * surface_albedo * cos_theta_direct
                    accumulated_color += path_throughput * direct_light_val
        path_throughput *= surface_albedo 
        if bounce_idx < MAX_BOUNCES - 1 : 
            if path_throughput.norm_sqr() < EPS*EPS: break
            if surface_normal_world.norm_sqr() < EPS*EPS: break 
            new_bounce_dir = sample_cosine_weighted_hemisphere_direction(surface_normal_world)
            if new_bounce_dir.norm_sqr() < EPS*EPS : break 
            current_ray_origin = hit_point_world + surface_normal_world * EPS * 10.0 
            current_ray_dir = new_bounce_dir
        else: break 
    return accumulated_color

@ti.kernel
def render(time_param: ti.f32, retina_z_slice_offset: ti.f32):
    cam_origin_world = tm.vec4(0.0, 0.0, 0.0, CAMERA_EYE_W_COORD)
    rot_xw = create_rotation_xw(time_param * 0.4); rot_yz = create_rotation_yz(time_param * 0.25)
    rot_xy = create_rotation_xy(time_param * 0.15); rot_zw = create_rotation_zw(time_param * 0.3)
    object_rotation_matrix = rot_xw @ rot_yz @ rot_xy @ rot_zw
    inv_object_rotation_matrix = object_rotation_matrix.transpose()
    for i, j in pixels:
        jitter_x = ti.random(ti.f32) - 0.5 
        jitter_y = ti.random(ti.f32) - 0.5
        u = (2.0 * (i + 0.5 + jitter_x) / _WIDTH - 1.0) * ASPECT_RATIO 
        v = 1.0 - (2.0 * (j + 0.5 + jitter_y) / _HEIGHT)             
        target_on_focal_plane = tm.vec4(u, v, retina_z_slice_offset, 0.0)
        initial_ray_dir = tm.normalize(target_on_focal_plane - cam_origin_world)
        final_color = trace_path(
            cam_origin_world, initial_ray_dir,
            object_rotation_matrix, inv_object_rotation_matrix)
        pixels[i, j] = tm.clamp(final_color, 0.0, 1.0)

# --- Upscale, Accumulation, Averaging Kernels ---
@ti.kernel
def upscale_nearest_neighbor(
    target_high_res: ti.template(), source_low_res: ti.template(),
    scale_factor_w: ti.f32, scale_factor_h: ti.f32 ):
    for i, j in target_high_res: 
        src_i = ti.cast(i / scale_factor_w, ti.i32); src_j = ti.cast(j / scale_factor_h, ti.i32)
        src_i = tm.clamp(src_i, 0, source_low_res.shape[0] - 1)
        src_j = tm.clamp(src_j, 0, source_low_res.shape[1] - 1)
        target_high_res[i, j] = source_low_res[src_i, src_j]
@ti.kernel
def average_accumulated_low_res(target_avg_low_res: ti.template(), source_accum_low_res: ti.template(), num_accum: ti.i32):
    for I in ti.grouped(target_avg_low_res):
        if num_accum > 0: target_avg_low_res[I] = source_accum_low_res[I] / num_accum
        else: target_avg_low_res[I] = source_accum_low_res[I] # Should not happen with num_accum >= 1
@ti.kernel
def copy_to_low_res_accumulation(target_accum_low_res: ti.template(), source_pixels_low_res: ti.template()):
    for I in ti.grouped(target_accum_low_res): target_accum_low_res[I] = source_pixels_low_res[I]
@ti.kernel
def add_to_low_res_accumulation(target_accum_low_res: ti.template(), source_pixels_low_res: ti.template()):
    for I in ti.grouped(target_accum_low_res): target_accum_low_res[I] += source_pixels_low_res[I]
@ti.kernel
def reset_low_res_accumulation_buffer(target_accum_low_res: ti.template()):
    for I in ti.grouped(target_accum_low_res): target_accum_low_res[I] = tm.vec3(0.0)


# --- Main Video Rendering Loop ---
if __name__ == "__main__":
    # Create temporary directory for frames
    if not os.path.exists(TEMP_FRAME_DIR):
        os.makedirs(TEMP_FRAME_DIR)

    # Generate retina z-slice offsets and weights programmatically
    num_slices = 32  # You can easily adjust this number as needed
    retina_z_slice_offset_list, retina_z_slice_offset_weights = generate_retina_slices(num_slices)
    
    # Print the generated values for verification
    print(f"Generated {num_slices} z-slices:")
    print(f"Offsets: {[round(x, 3) for x in retina_z_slice_offset_list]}")
    print(f"Weights: {[round(x, 3) for x in retina_z_slice_offset_weights]}")
    
    retina_z_slice_offset = 0.0  # Default value

    total_animation_duration = VIDEO_TOTAL_FRAMES / VIDEO_FPS
        
    for video_frame_idx in range(VIDEO_TOTAL_FRAMES):
        frame_images = []

        current_animation_time = (video_frame_idx / VIDEO_TOTAL_FRAMES) * total_animation_duration
        print(f"Rendering Video Frame: {video_frame_idx + 1}/{VIDEO_TOTAL_FRAMES} (Animation Time: {current_animation_time:.2f}s)")

        for i in range(len(retina_z_slice_offset_list)):
            retina_z_slice_offset = retina_z_slice_offset_list[i]
            print(f"  ... rendering slice {i+1}/{len(retina_z_slice_offset_list)} (z-slice offset: {retina_z_slice_offset:.2f})")

            reset_low_res_accumulation_buffer(accumulated_pixels)
            num_samples_for_this_frame = 0
            
            accumulation_start_time = time.monotonic()
            while True:
                render(current_animation_time, retina_z_slice_offset) # Render the static scene for this animation time

                if num_samples_for_this_frame == 0:
                    copy_to_low_res_accumulation(accumulated_pixels, pixels)
                else:
                    add_to_low_res_accumulation(accumulated_pixels, pixels)
                num_samples_for_this_frame += 1

                elapsed_accumulation_time = time.monotonic() - accumulation_start_time
                if elapsed_accumulation_time >= ACCUMULATION_SECONDS_PER_VIDEO_FRAME:
                    break
            
            average_accumulated_low_res(averaged_low_res_image, accumulated_pixels, num_samples_for_this_frame)
            print(f"  ... accumulated {num_samples_for_this_frame} samples in {elapsed_accumulation_time:.2f}s.")

            # Prepare frame for saving
            if SAVE_UPSCALED_FRAMES:
                upscale_nearest_neighbor(frame_save_buffer, averaged_low_res_image,
                                        float(GUI_WIDTH)/RENDER_WIDTH, float(GUI_HEIGHT)/RENDER_HEIGHT)
                img_to_save_np = frame_save_buffer.to_numpy()
            else:
                img_to_save_np = averaged_low_res_image.to_numpy()

            # Convert to BGR for OpenCV and scale to 0-255
            img_to_save_np = (np.clip(img_to_save_np, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_to_save_np, cv2.COLOR_RGB2BGR)
            
            # OpenCV expects height, width, channels
            # Taichi to_numpy() gives width, height, channels. Need to transpose for image orientation.
            # However, if ASPECT_RATIO = 9/16 (portrait), RENDER_WIDTH < RENDER_HEIGHT.
            # If to_numpy() outputs (width, height, channels), and cv2 needs (height, width, channels) for an image,
            # we need to be careful.
            # Let's assume to_numpy() gives (dim1, dim2, channels) where dim1 corresponds to shape[0], dim2 to shape[1]
            # For Taichi field(shape=(W,H)), to_numpy() is (W,H,C).
            # cv2.imwrite expects image as (H,W,C). So transpose (0,1) -> (1,0)
            if img_bgr.shape[0] == _WIDTH and img_bgr.shape[1] == _HEIGHT: # If it's W,H,C
                img_bgr_transposed = np.transpose(img_bgr, (1, 0, 2))
            elif SAVE_UPSCALED_FRAMES and img_bgr.shape[0] == GUI_WIDTH and img_bgr.shape[1] == GUI_HEIGHT:
                img_bgr_transposed = np.transpose(img_bgr, (1, 0, 2))
            else: # If it's already H,W,C (less likely from Taichi default)
                img_bgr_transposed = img_bgr

            frame_images.append(img_bgr_transposed)


        # Combine all images for this frame into a single image weighted by the normalized weights
        combined_image = np.zeros_like(frame_images[0], dtype=np.float32)
        for i in range(len(frame_images)):
            combined_image += frame_images[i] * retina_z_slice_offset_weights[i]

        combined_image /= sum(retina_z_slice_offset_weights)
        combined_image = np.clip(combined_image, 0, 255)
        combined_image = combined_image.astype(np.uint8)

        frame_filename = os.path.join(TEMP_FRAME_DIR, f"frame_{video_frame_idx:04d}.png")
        cv2.imwrite(frame_filename, combined_image)

    print(f"All {VIDEO_TOTAL_FRAMES} frames saved to {TEMP_FRAME_DIR}/")

