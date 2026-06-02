import open3d as o3d
import numpy as np
import os
from typing import List, Tuple

class CameraPlanner:
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    @staticmethod
    def _sample_points_on_sphere(center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
        """Sample coverage points on a sphere surface with Fibonacci sampling."""
        if num_points <= 0:
            return np.array([])
            
        
        PHI = (1 + np.sqrt(5)) / 2
        
        points = []
        for i in range(num_points):
            
            lon = 2 * np.pi * (i / PHI) % (2 * np.pi)
            
            lat = np.arccos(1 - 2 * (i + 0.5) / num_points)
            
            x = radius * np.sin(lat) * np.cos(lon)
            y = radius * np.sin(lat) * np.sin(lon)
            z = radius * np.cos(lat)
            
            points.append([x, y, z])
            
        return center + np.array(points)

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def __init__(self,args):
        self.args = args
        self.init_mesh(args["point_path"],args["mesh_path"])
        
        self.sphere_points_np = np.array([])
        
        self.sphere_coverage_status = np.array([], dtype=np.uint8)
        self.AREA_PER_POINT = 0.0 

        self.sphere_points_view = np.array([]) 
        self.ray_casting_scene = None 

    def init_mesh(self, input_ply_path, output_ply_path, voxel_size=0.005, nb_neighbors=20, std_ratio=2.0, normal_radius=1.5, normal_max_nn=30, poisson_depth=9,gui=False,is_crop=True):
        """Process a point cloud and run Poisson reconstruction to build a mesh."""
        
        print("--- 1. Loading point cloud ---")
        if not os.path.exists(input_ply_path):
            print(f"Error: file not found: {input_ply_path}")
            
            self.pcd_bbox = o3d.geometry.OrientedBoundingBox() 
            self.mesh = o3d.geometry.TriangleMesh()
            return
            
        pcd = o3d.io.read_point_cloud(input_ply_path)
        print(f"Raw point cloud has {len(pcd.points)} points")

        
        if gui:
            o3d.visualization.draw_geometries([pcd], window_name="Raw point cloud")
        
        print("\n--- 2. Preprocessing point cloud ---")
        
        
        print("Removing outliers...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd_filtered = pcd.select_by_index(ind)
        print(f"Point cloud has {len(pcd_filtered.points)} points after outlier removal")
        
        
        print("Running voxel downsampling...")
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
        print(f"Point cloud has {len(pcd_downsampled.points)} points after downsampling")
        
        
        if len(pcd_downsampled.points) == 0:
             print("Point cloud is empty; skipping reconstruction.")
             self.pcd_bbox = o3d.geometry.OrientedBoundingBox() 
             self.mesh = o3d.geometry.TriangleMesh()
             return

        self.pcd_bbox = pcd_downsampled.get_oriented_bounding_box()
        if is_crop:
            print("Cropping point cloud to its oriented bounding box...")
            pcd_downsampled = pcd_downsampled.crop(self.pcd_bbox)


        print("\n--- 3. Estimating and orienting normals ---")
        
        
        print("Estimating point cloud normals...")
        pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        
        
        print("Orienting normals consistently...")
        pcd_downsampled.orient_normals_consistent_tangent_plane(k=30)
        
        print("\n--- 4. Running Poisson reconstruction ---")
        
        
        print(f"Running Poisson reconstruction with depth={poisson_depth}...")
        try:
             mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_downsampled, depth=poisson_depth)
        except:
             print("Poisson reconstruction failed, likely due to normals; returning an empty mesh.")
             self.mesh = o3d.geometry.TriangleMesh()
             o3d.io.write_triangle_mesh(output_ply_path, self.mesh)
             return


        print("\n--- 5. Post-processing mesh ---")
        
        
        print("Removing low-density vertices...")
        densities_np = np.asarray(densities)
        
        vertices_to_remove = densities_np < np.quantile(densities_np, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print("Cropping mesh to the oriented bounding box...")
        mesh_cropped = mesh.crop(self.pcd_bbox)
        
        print(
            f"Reconstruction complete: {len(mesh_cropped.vertices)} vertices, "
            f"{len(mesh_cropped.triangles)} triangles"
        )
        
        
        if gui:
            print("Visualizing cropped mesh and OBB...")
            o3d.visualization.draw_geometries([mesh_cropped, self.pcd_bbox], window_name="Cropped mesh and OBB")

        
        o3d.io.write_triangle_mesh(output_ply_path, mesh_cropped)
        print(f"\nMesh saved to {output_ply_path}")
        self.mesh = mesh_cropped

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def sample_spheres(self, is_gui=False):
        def load_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
            """Load a PLY file as an Open3D TriangleMesh."""
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                if not mesh.has_vertices() or not mesh.has_triangles():
                    print(f"Warning: {file_path} loaded but has no valid vertices or triangles.")
                    return None
                print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                return None

        
        def sample_points_to_cover_mesh(mesh: o3d.geometry.TriangleMesh, voxel_size: float, sphere_radius: float) -> o3d.geometry.PointCloud:
            """Sample sphere centers from the mesh surface."""
            print("Generating dense initial point cloud...")
            
            num_points = max(1000000, int(len(mesh.vertices) * 10)) 
            dense_pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            pcd_samples = dense_pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"Sampled {len(pcd_samples.points)} sphere centers from the mesh")
            return pcd_samples

        
        def sample_points_on_obb_faces(obb: o3d.geometry.OrientedBoundingBox, point_spacing: float) -> o3d.geometry.PointCloud:
            """Sample sphere centers uniformly on the six OBB faces."""
            print("\n--- Sampling sphere centers on OBB faces ---")
            
            center = obb.center
            R = obb.R
            extent = obb.extent
            X_local = R[:, 0]; Y_local = R[:, 1]; Z_local = R[:, 2]
            face_points = []
            
            
            step_x = max(point_spacing, extent[0] / 50)
            step_y = max(point_spacing, extent[1] / 50)
            step_z = max(point_spacing, extent[2] / 50)

            
            for z_sign in [-1, 1]:
                z_offset = z_sign * extent[2] / 2 * Z_local
                for i in np.arange(-extent[0] / 2, extent[0] / 2 + 1e-6, step_x):
                    for j in np.arange(-extent[1] / 2, extent[1] / 2 + 1e-6, step_y):
                        face_points.append(center + i * X_local + j * Y_local + z_offset)

            
            for y_sign in [-1, 1]:
                y_offset = y_sign * extent[1] / 2 * Y_local
                for i in np.arange(-extent[0] / 2 + step_x/2, extent[0] / 2 - step_x/2 + 1e-6, step_x):
                    for k in np.arange(-extent[2] / 2 + step_z/2, extent[2] / 2 - step_z/2 + 1e-6, step_z):
                        face_points.append(center + i * X_local + y_offset + k * Z_local)

            
            for x_sign in [-1, 1]:
                x_offset = x_sign * extent[0] / 2 * X_local
                for j in np.arange(-extent[1] / 2 + step_y/2, extent[1] / 2 - step_y/2 + 1e-6, step_y):
                    for k in np.arange(-extent[2] / 2 + step_z/2, extent[2] / 2 - step_z/2 + 1e-6, step_z):
                        face_points.append(center + x_offset + j * Y_local + k * Z_local)

            obb_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array(face_points)))
            print(f"Sampled {len(obb_pcd.points)} sphere centers from the OBB")
            return obb_pcd

        # -------------------------------------------------------------------
        
        # -------------------------------------------------------------------
        
        mesh = self.mesh 
        if mesh is None or not mesh.has_vertices(): 
            if not hasattr(self, 'pcd_bbox'):
                print("Unable to load mesh and no OBB is available; stopping sampling.")
                return
            else:
                print("Mesh is unavailable; sampling sphere centers from the OBB only.")

        
        
        sphere_radius = self.args["sphere_radius"]
        voxel_size = 2 * sphere_radius * 0.9 
        
        
        NUM_POINTS_PER_SPHERE = self.args.get("num_points_per_sphere", 1000)

        
        sphere_total_area = 4.0 * np.pi * sphere_radius**2
        self.AREA_PER_POINT = sphere_total_area / NUM_POINTS_PER_SPHERE
        print(f"Coverage area per subdivided point: {self.AREA_PER_POINT:.6f} m^2")


        
        mesh_center_points_np = np.array([])
        mesh_center_count = 0 
        if mesh is not None and len(mesh.vertices) > 0:
            pcd_mesh_samples = sample_points_to_cover_mesh(mesh, voxel_size, sphere_radius)
            mesh_center_points_np = np.asarray(pcd_mesh_samples.points)
            mesh_center_count = len(mesh_center_points_np) 
            
        obb_center_points_np = np.array([])
        if hasattr(self, 'pcd_bbox') and np.linalg.norm(self.pcd_bbox.extent) > 1e-6:
            pcd_obb_samples = sample_points_on_obb_faces(self.pcd_bbox, voxel_size)
            
            
            obb_center_points_np = np.asarray(pcd_obb_samples.points)
        
        if len(mesh_center_points_np) > 0 or len(obb_center_points_np) > 0:
            all_center_points_np = np.concatenate([mesh_center_points_np, obb_center_points_np], axis=0)
        else:
             all_center_points_np = np.array([])
             print("No sphere centers were sampled; stopping.")
             return

        
        print(
            f"\n--- Subdividing {len(all_center_points_np)} spheres "
            f"into {NUM_POINTS_PER_SPHERE} coverage points each ---"
        )
        
        all_coverage_points = []
        
        mesh_point_indices = []
        
        current_global_index = 0
        for idx, center in enumerate(all_center_points_np):
            points_on_sphere = self._sample_points_on_sphere(center, sphere_radius, NUM_POINTS_PER_SPHERE)
            all_coverage_points.append(points_on_sphere)
            
            
            if idx < mesh_center_count:
                
                mesh_point_indices.extend(range(current_global_index, current_global_index + NUM_POINTS_PER_SPHERE))
                
            current_global_index += NUM_POINTS_PER_SPHERE
            
        self.sphere_points_np = np.concatenate(all_coverage_points, axis=0)
        self.sphere_coverage_status = np.zeros(len(self.sphere_points_np), dtype=np.uint8)
        
        
        self.mesh_point_indices = np.array(mesh_point_indices, dtype=np.int64)
        print(
            f"Mesh-surface coverage points: {len(self.mesh_point_indices)}. "
            f"Total coverage points: {len(self.sphere_points_np)}"
        )
        
        
        self.ray_casting_scene = None
        if mesh is not None and mesh.has_triangles():
            try:
                self.ray_casting_scene = o3d.t.geometry.RaycastingScene()
                
                tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                
                self.ray_casting_scene.add_triangles(tensor_mesh)
                print("RayCastingScene initialized successfully.")
            except Exception as e:
                print(f"Warning: RaycastingScene initialization failed ({e}); occlusion culling will be skipped.")
                self.ray_casting_scene = None
    
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def _get_frustum_visibility(self, M_c2w: np.ndarray, window_width: int, window_height: int, intrinsic_matrix: np.ndarray, max_depth: float = 1.0) -> np.ndarray:
        """Check which coverage points fall inside the camera frustum without occlusion testing."""
        sphere_points_np = self.sphere_points_np
        if len(sphere_points_np) == 0:
            return np.array([], dtype=bool)

        
        M_w2c = np.linalg.inv(M_c2w)
        points_hom = np.hstack([sphere_points_np, np.ones((len(sphere_points_np), 1))])
        points_cam_hom = (M_w2c @ points_hom.T).T
        points_cam = points_cam_hom[:, :3]
        
        
        is_in_front = points_cam[:, 2] > 1e-6 
        is_within_far_clip = np.full(len(sphere_points_np), True)
        if max_depth > 0:
            is_within_far_clip = points_cam[:, 2] < max_depth
            
        
        K = intrinsic_matrix
        z_c = points_cam[:, 2]
        safe_z_c = np.where(z_c > 1e-6, z_c, 1e-6)
        
        x_norm = points_cam[:, 0] / safe_z_c
        y_norm = points_cam[:, 1] / safe_z_c
        
        u = K[0, 0] * x_norm + K[0, 2]
        v = K[1, 1] * y_norm + K[1, 2]

        
        is_on_screen_u = (u >= 0) & (u < window_width)
        is_on_screen_v = (v >= 0) & (v < window_height)
        
        
        is_visible_frustum = is_in_front & is_on_screen_u & is_on_screen_v & is_within_far_clip
        
        return is_visible_frustum

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def _get_mesh_only_visibility(self, M_c2w: np.ndarray, window_width: int, window_height: int, intrinsic_matrix: np.ndarray, max_depth: float = 1.0) -> np.ndarray:
        """
        Check which mesh-surface samples are inside the frustum and not occluded.

        Returns:
            A boolean mask with the same length as self.sphere_points_np.
            Only samples from mesh surfaces can be marked visible.
        """
        
        
        final_visibility = np.zeros(len(self.sphere_points_np), dtype=bool)

        if len(self.mesh_point_indices) == 0:
            return final_visibility 

        
        mesh_points_np = self.sphere_points_np[self.mesh_point_indices]
        
        
        
        original_sphere_points = self.sphere_points_np
        self.sphere_points_np = mesh_points_np
        is_visible_frustum_mesh = self._get_frustum_visibility(M_c2w, window_width, window_height, intrinsic_matrix, max_depth)
        self.sphere_points_np = original_sphere_points 
        
        
        frustum_visible_mesh_indices = np.where(is_visible_frustum_mesh)[0]
        frustum_visible_points = mesh_points_np[frustum_visible_mesh_indices]
        
        if len(frustum_visible_points) == 0:
            return final_visibility 

        
        if self.ray_casting_scene is None:
            
            
            global_indices_to_set_true = self.mesh_point_indices[frustum_visible_mesh_indices]
            final_visibility[global_indices_to_set_true] = True
            return final_visibility
            
        
        camera_origin = M_c2w[:3, 3]
        
        
        ray_directions = frustum_visible_points - camera_origin
        expected_distances = np.linalg.norm(ray_directions, axis=1)
        ray_directions_normalized = ray_directions / expected_distances[:, np.newaxis]
        
        
        rays = np.hstack([
            np.tile(camera_origin, (len(frustum_visible_points), 1)), # Origin
            ray_directions_normalized  # Direction
        ])
        
        
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        try:
            ans = self.ray_casting_scene.cast_rays(rays_tensor)
        except Exception as e:
             print(f"Raycasting failed: {e}. Falling back to frustum visibility.")
             global_indices_to_set_true = self.mesh_point_indices[frustum_visible_mesh_indices]
             final_visibility[global_indices_to_set_true] = True
             return final_visibility

        hit_distances = ans['t_hit'].numpy()
        
        
        is_occluded_by_mesh = hit_distances < (expected_distances - 1e-3)
        is_not_occluded = ~is_occluded_by_mesh
        
        
        
        global_indices_of_all_frustum_points = self.mesh_point_indices[frustum_visible_mesh_indices]
        global_indices_of_visible_points = global_indices_of_all_frustum_points[is_not_occluded]

        
        final_visibility[global_indices_of_visible_points] = True
        
        return final_visibility
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def visualize_mesh_coverage_only(self, M_c2w: np.ndarray, K: np.ndarray, title: str = "Mesh Surface Coverage (Visible/Occluded)"):
        """
        Visualize mesh samples visible or occluded from a single camera pose.

        Blue points are visible mesh samples, red points are occluded mesh
        samples, and gray points are all other samples.
        """
        if not hasattr(self, 'mesh_point_indices') or len(self.sphere_points_np) == 0:
            print("No samples to visualize. Run sample_spheres first and check mesh_point_indices.")
            return
            
        print(f"\n--- Visualizing mesh-only coverage: {title} ---")
        
        window_width = int(K[0, 2] * 2)
        window_height = int(K[1, 2] * 2)

        
        
        
        is_visible_mesh_only = self._get_mesh_only_visibility(M_c2w, window_width, window_height, K)
        
        
        is_visible_frustum = self._get_frustum_visibility(M_c2w, window_width, window_height, K)
        
        
        pcd_all_points = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.sphere_points_np))
        colors = np.zeros_like(self.sphere_points_np)
        
        
        VISIBLE_COLOR = np.array([0.0, 0.0, 1.0])  
        OCCLUDED_COLOR = np.array([1.0, 0.0, 0.0]) 
        OTHER_COLOR = np.array([0.5, 0.5, 0.5])    

        
        colors[:] = OTHER_COLOR 

        
        mesh_indices = self.mesh_point_indices
        
        
        
        is_occluded_mesh_points = (is_visible_frustum & (~is_visible_mesh_only))
        
        
        
        
        colors[is_occluded_mesh_points] = OCCLUDED_COLOR
        
        
        colors[is_visible_mesh_only] = VISIBLE_COLOR 

        pcd_all_points.colors = o3d.utility.Vector3dVector(colors)
        
        
        frustum = self._create_camera_frustum_lineset(M_c2w, K, color=(1, 0, 0)) 
        
        
        geometries = []
        if hasattr(self, 'mesh') and self.mesh.has_vertices():
            self.mesh.paint_uniform_color([0.6, 0.6, 0.6]) 
            geometries.append(self.mesh) 
        if hasattr(self, 'pcd_bbox'):
            self.pcd_bbox.color = (0, 0, 1)
            geometries.append(self.pcd_bbox) 
            
        geometries.append(pcd_all_points)
        geometries.append(frustum)
        
        
        o3d.visualization.draw_geometries(geometries, window_name=title, point_show_normal=False)
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    
    def _get_visibility_with_raycasting(self, M_c2w: np.ndarray, window_width: int, window_height: int, intrinsic_matrix: np.ndarray, max_depth: float = 1.0) -> np.ndarray:
        """
        Check whether each sampled point is in the frustum and not mesh-occluded.

        Returns:
            A boolean mask indicating final visibility for each sampled point.
        """
        sphere_points_np = self.sphere_points_np
        if len(sphere_points_np) == 0:
            return np.array([], dtype=bool)

        
        is_visible_frustum = self._get_frustum_visibility(M_c2w, window_width, window_height, intrinsic_matrix, max_depth)
        
        
        frustum_visible_indices = np.where(is_visible_frustum)[0]
        frustum_visible_points = sphere_points_np[frustum_visible_indices]
        
        if len(frustum_visible_points) == 0:
            return is_visible_frustum 

        
        if self.ray_casting_scene is None:
            
            return is_visible_frustum
            
        

        
        camera_origin = M_c2w[:3, 3]
        
        
        ray_directions = frustum_visible_points - camera_origin
        
        expected_distances = np.linalg.norm(ray_directions, axis=1)
        
        ray_directions_normalized = ray_directions / expected_distances[:, np.newaxis]
        
        
        rays = np.hstack([
            np.tile(camera_origin, (len(frustum_visible_points), 1)), # Origin
            ray_directions_normalized                                 # Direction
        ])
        
        
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        try:
            ans = self.ray_casting_scene.cast_rays(rays_tensor)
        except Exception as e:
             
             
             return is_visible_frustum

        hit_distances = ans['t_hit'].numpy()
        
        
        
        
        is_occluded_by_mesh = hit_distances < (expected_distances - 1e-3)
        
        
        is_not_occluded = ~is_occluded_by_mesh
        
        
        final_visibility = np.zeros(len(sphere_points_np), dtype=bool)
        
        
        final_visibility[frustum_visible_indices[is_not_occluded]] = True
        
        
        
        return final_visibility

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def init_cover_sets(self,init_cams):
        cam_nums = len(init_cams)
        covers_sets = []
        for i in range(cam_nums):
            for j in range(i+1,cam_nums,1):
                M_c2w_i = init_cams[i]['camtoworlds']
                K_i = init_cams[i]['K']
                window_width_i = int(K_i[0, 2] * 2) 
                window_height_i = int(K_i[1, 2] * 2)
                
                is_visible_i = self.visualize_mesh_coverage_only(M_c2w_i, window_width_i, window_height_i, K_i,max_depth=-1)
                
                M_c2w_j = init_cams[j]['camtoworlds']
                K_j = init_cams[j]['K']
                window_width_j = int(K_j[0, 2] * 2) 
                window_height_j = int(K_j[1, 2] * 2)
                
                is_visible_j = self._get_visibility_with_raycasting(M_c2w_j, window_width_j, window_height_j, K_j,max_depth=-1)
                
                covers_sets.append({
                        'coverage': np.maximum(is_visible_i.astype(np.uint8), is_visible_j.astype(np.uint8)),
                        'id': [i,j]
                    })
        return covers_sets
    def determine_obb_z_sign(self, M_c2w: np.ndarray) -> int:
        """
        Determine whether the camera look-at axis aligns with the OBB Z axis.

        Returns +1 for the positive OBB Z direction and -1 for the negative
        direction.
        """
        if not hasattr(self, 'pcd_bbox') or np.linalg.norm(self.pcd_bbox.extent) < 1e-6:
            print("Warning: OBB is missing or degenerate. Falling back to Z sign 1.")
            return 1
            
        
        
        obb_rotation = self.pcd_bbox.R 
        obb_z_axis_world = obb_rotation[:, 2] 
        
        
        
        camera_z_axis_world = M_c2w[:3, 2] 
        
        
        obb_z_axis_world = obb_z_axis_world / np.linalg.norm(obb_z_axis_world)
        camera_z_axis_world = camera_z_axis_world / np.linalg.norm(camera_z_axis_world)

        
        dot_product = np.dot(camera_z_axis_world, obb_z_axis_world)
        
        
        
        if dot_product >= 0:
            
            return 1
        else:
            
            return -1
    
    def init_axis(self,K_i):
        point_end_param = [0, 0, 0] 
        point_start_param = [0, 0, 1]
        point_start_param2 = [0, 0, -1]
        camtoworlds_i = self.create_cameras(point_start_param, point_end_param,1)
        window_width_i = int(K_i[0, 2] * 2) 
        window_height_i = int(K_i[1, 2] * 2)
        is_visible_i = np.sum(self._get_mesh_only_visibility(camtoworlds_i, window_width_i, window_height_i, K_i,max_depth=-1).astype(np.uint8))
        camtoworlds_i = self.create_cameras(point_start_param2, point_end_param,1)
        window_width_i = int(K_i[0, 2] * 2) 
        window_height_i = int(K_i[1, 2] * 2)
        is_visible_j = np.sum(self._get_mesh_only_visibility(camtoworlds_i, window_width_i, window_height_i, K_i,max_depth=-1).astype(np.uint8))
        if is_visible_i>is_visible_j:
            return -1
        else:
            return 1
            
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def evaluate_camera_view_whole(self, M_c2w_list: List[np.ndarray], K_list: List[np.ndarray], coverage_threshold: float, is_gui=False) -> Tuple[bool, float]:
        """
        Evaluate a batch of camera poses using the exact union of sample coverage.

        Returns:
            Whether the batch is accepted and its newly covered surface area.
        """
        if not hasattr(self, 'sphere_coverage_status') or len(self.sphere_points_np) == 0:
            return False, 0.0

        print(
            f"\n--- Evaluating {len(M_c2w_list)} candidate camera poses "
            f"against batch area threshold {coverage_threshold:.4f} m^2 ---"
        )
        
        total_points = len(self.sphere_points_np)
        current_covered_points = np.sum(self.sphere_coverage_status)
        
        
        

        
        
        temp_batch_coverage = np.copy(self.sphere_coverage_status)
        
        
        for i, (M_c2w, K) in enumerate(zip(M_c2w_list, K_list)):
            
            window_width = int(K[0, 2] * 2) if K is not None else 800
            window_height = int(K[1, 2] * 2) if K is not None else 800
            
            
            is_visible = self._get_visibility_with_raycasting(M_c2w, window_width, window_height, K)
            
            
            temp_batch_coverage = np.maximum(temp_batch_coverage, is_visible.astype(np.uint8))
        
        
        
        points_covered_after_batch = np.sum(temp_batch_coverage)
        newly_covered_points = points_covered_after_batch - current_covered_points
        
        
        total_newly_covered_area = newly_covered_points * self.AREA_PER_POINT
        
        print(f"\nBatch newly covered samples: {newly_covered_points}.")
        print(f"Batch newly covered area: {total_newly_covered_area:.4f} m^2.")

        
        if total_newly_covered_area >= coverage_threshold:
            print(
                f"Batch accepted: new area {total_newly_covered_area:.4f} m^2 "
                f">= threshold {coverage_threshold:.4f} m^2."
            )
            
            
            self.sphere_coverage_status = temp_batch_coverage
            
            return True, total_newly_covered_area
        else:
            print(
                f"Batch rejected: new area {total_newly_covered_area:.4f} m^2 "
                f"< threshold {coverage_threshold:.4f} m^2."
            )
            return False, 0.0

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def create_cameras(self,point_start, point_end,z_axis):
        
        obb = self.pcd_bbox
        obb_center = obb.center 
        obb_rotation = obb.R 
        obb_extent = obb.extent 

        
        local_x_axis_world = obb_rotation @ np.array([1, 0, 0])
        local_y_axis_world = obb_rotation @ np.array([0, 1, 0])
        local_z_axis_world = obb_rotation @ np.array([0, 0, 1.0])

        
        P_end1 = obb_center + point_start[0]*local_x_axis_world * (obb_extent[0] / 2) + point_start[1]*local_y_axis_world * (obb_extent[1] / 2) + point_start[2]*local_z_axis_world * (obb_extent[2] / 2)
        P_end2 = obb_center + point_end[0]*local_x_axis_world * (obb_extent[0] / 2) + point_end[1]*local_y_axis_world * (obb_extent[1] / 2) + point_end[2]*local_z_axis_world * (obb_extent[2] / 2)
        
        
        camera_position = P_end1 
        camera_lookat = P_end2 

        
        
        
        camera_front = camera_lookat - camera_position
        camera_front = camera_front / np.linalg.norm(camera_front)

        
        original_up = obb_rotation @ np.array([0, 0, 1*z_axis]) 
        
        
        
        Z_c = camera_front
        
        
        Y_temp = original_up
        
        
        X_c = np.cross(Y_temp, Z_c)
        
        if np.linalg.norm(X_c) < 1e-6:
             
             Y_temp = np.array([0, 1, 0]) if np.abs(np.dot(Z_c, np.array([0, 1, 0]))) < 0.9 else np.array([1, 0, 0])
             X_c = np.cross(Y_temp, Z_c)

        X_c = X_c / np.linalg.norm(X_c)

        
        Y_c = np.cross(Z_c, X_c)
        
        
        
        R_c2w = np.stack([X_c, Y_c, Z_c], axis=1) 
        
        M_c2w = np.identity(4)
        M_c2w[:3, :3] = R_c2w
        M_c2w[:3, 3] = camera_position 
        return M_c2w

    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    def visualize_coverage(self, camera_to_add: o3d.geometry.LineSet = None, title="Current Coverage State"):
        """
        Visualize the scene mesh, OBB, and sampled-point coverage state.
        
        Args:
            camera_to_add: Optional camera frustum geometry to draw with the scene.
        """
        if len(self.sphere_points_np) == 0:
            print("No samples to visualize. Run sample_spheres first.")
            return
            
        print(f"\n--- Visualizing coverage state: {title} ---")
        geometries = []
        
        
        if hasattr(self, 'mesh') and self.mesh.has_vertices():
            self.mesh.paint_uniform_color([0.6, 0.6, 0.6])
            geometries.append(self.mesh)
        if hasattr(self, 'pcd_bbox'):
            self.pcd_bbox.color = (0, 0, 1) 
            geometries.append(self.pcd_bbox)
        
        
        pcd_coverage = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.sphere_points_np))
        
        colors = np.zeros_like(self.sphere_points_np)
        
        
        COVERED_COLOR = np.array([0.0, 1.0, 0.0])  
        UNCOVERED_COLOR = np.array([1.0, 0.0, 0.0]) 

        
        is_covered = self.sphere_coverage_status == 1
        colors[is_covered] = COVERED_COLOR
        colors[~is_covered] = UNCOVERED_COLOR
        
        pcd_coverage.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd_coverage)
        
        
        if camera_to_add is not None:
            geometries.append(camera_to_add)

        
        o3d.visualization.draw_geometries(geometries, 
                                          window_name=title,
                                          point_show_normal=False)


    def _create_camera_frustum_lineset(self, M_c2w: np.ndarray, K: np.ndarray, color: Tuple[float, float, float] = (0, 0, 1)) -> o3d.geometry.LineSet:
        """
        Create a LineSet geometry for the camera frustum.
        """
        if K is None:
             print("Warning: missing intrinsics K; cannot draw the frustum.")
             return o3d.geometry.LineSet()

        H = int(K[1, 2] * 2)
        W = int(K[0, 2] * 2)
        
        
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        
        near = 0.1 
        
        
        corners_cam = np.array([
            [ (0 - cx) / fx * near, (0 - cy) / fy * near, near ], # Top-Left (0, 0)
            [ (W - cx) / fx * near, (0 - cy) / fy * near, near ], # Top-Right (W, 0)
            [ (W - cx) / fx * near, (H - cy) / fy * near, near ], # Bottom-Right (W, H)
            [ (0 - cx) / fx * near, (H - cy) / fy * near, near ], # Bottom-Left (0, H)
        ])
        
        
        R_c2w = M_c2w[:3, :3]
        t_c2w = M_c2w[:3, 3]
        
        
        center_w = t_c2w 
        
        
        corners_w = (R_c2w @ corners_cam.T).T + t_c2w
        
        
        points = [center_w] + [corners_w[i] for i in range(4)]
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4], 
            [1, 2], [2, 3], [3, 4], [4, 1]  
        ]
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(points)),
            lines=o3d.utility.Vector2iVector(np.array(lines))
        )
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        
        return line_set


    def visualize_camera_view(self, M_c2w: np.ndarray, K: np.ndarray, title: str = "Camera View and Visible Points"):
        """
        Visualize a single camera pose and its visible sampled points.
        """
        if len(self.sphere_points_np) == 0:
            print("No samples to visualize. Run sample_spheres first.")
            return
            
        print(f"\n--- Visualizing single camera view: {title} ---")
        
        window_width = int(K[0, 2] * 2)
        window_height = int(K[1, 2] * 2)

        
        is_visible = self._get_visibility_with_raycasting(M_c2w, window_width, window_height, K)
        visible_points = self.sphere_points_np[is_visible]
        
        
        pcd_visible = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(visible_points))
        pcd_visible.paint_uniform_color([0.0, 0.0, 1.0]) 

        
        frustum = self._create_camera_frustum_lineset(M_c2w, K, color=(1, 0, 0)) 
        
        
        geometries = []
        if hasattr(self, 'mesh') and self.mesh.has_vertices():
            self.mesh.paint_uniform_color([0.6, 0.6, 0.6]) 
            geometries.append(self.mesh) 
        if hasattr(self, 'pcd_bbox'):
            self.pcd_bbox.color = (0, 0, 1)
            geometries.append(self.pcd_bbox) 
            
        geometries.append(pcd_visible)
        geometries.append(frustum)
        
        
        o3d.visualization.draw_geometries(geometries, window_name=title, point_show_normal=False)
Cam_planer = CameraPlanner
