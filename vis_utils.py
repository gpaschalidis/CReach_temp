import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import os


def visualize_rays_after_scanning(clean_mask, i,new_origins,targets,new_clean_points, obj_center,obj_mesh,rec_mesh,mesh_box):
    clean_mask = clean_mask.reshape(-1, 5,10)
    clean_mask = clean_mask[:,:,0:i+1].reshape(-1,5*(i+1))
    final_mask = clean_mask.sum(1) == 5*(i+1)
    final_clean_points = new_clean_points[final_mask]
    #new_origins = new_origins[:,:,0:i+1,:][final_mask]
    new_origins = new_origins[final_mask]
    #targets = targets[:,:,0:i+1,:][final_mask]
    targets = targets[final_mask]
    line_set_new = create_line_set(new_origins.reshape(-1,3), targets.reshape(-1,3), [0.5, 0.1, 0.2], multiple_origins=True)
    line_set_new_4 = create_line_set(obj_center[None], final_clean_points, [1, 0, 1])
    o3d.visualization.draw_geometries([rec_mesh, obj_mesh, mesh_box, line_set_new,line_set_new_4])


def define_obj_key(obj_name, list_per_obj, rec_key):
    comb = list_per_obj[0]
    comb_parts = comb.split("_")
    obj_where = "_".join((comb_parts[1],comb_parts[2]))
    obj_key = "_".join((obj_name, rec_key, obj_where))
    return obj_key


def sphere_around_obj(obj_height, obj_center):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2*obj_height)
    sp_verts = np.array(sphere.vertices)
    sp_verts = sp_verts + obj_center
    return sp_verts

    
def read_o3d_mesh(path):
    return o3d.io.read_triangle_mesh(path)


def create_o3d_mesh(verts, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_o3d_box_mesh(rec_verts):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=4,height=4,depth=0.005)
    mesh_box_verts = np.array(mesh_box.vertices)
    mesh_box_center = (mesh_box_verts.max(0)+mesh_box_verts.min(0))/2
    mesh_box_verts -= mesh_box_center
    mesh_box_verts[:,0] += rec_verts.mean(0)[0]
    mesh_box_verts[:,1] += rec_verts.mean(0)[1]
    mesh_box.vertices = o3d.utility.Vector3dVector(mesh_box_verts)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.2, 0.2])
    return mesh_box


def create_o3d_box_mesh_vertical_y(rec_verts):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=4,height=4,depth=0.005)
    mesh_box_verts = np.array(mesh_box.vertices)
    mesh_box_center = (mesh_box_verts.max(0)+mesh_box_verts.min(0))/2
    mesh_box_verts -= mesh_box_center
    Rx = np.eye(3)
    Rx[1][1] = 0 
    Rx[2][2] = 0 
    Rx[1][2] = -1
    Rx[2][1] = 1 
    mesh_box_verts = (Rx.T @ mesh_box_verts.T).T

    mesh_box_verts[:,0] += rec_verts.mean(0)[0]
    mesh_box_verts[:,2] += rec_verts.mean(0)[2]
    mesh_box.vertices = o3d.utility.Vector3dVector(mesh_box_verts)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.2, 0.2])
    return mesh_box

def make_background_white(o3d_image):
    o3d_image = np.array(o3d_image)
    mask = o3d_image >= 234
    o3d_image[mask] = 255
    o3d_image = o3d.geometry.Image(o3d_image)
    return o3d_image


def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(list(np.random.rand(1,3)[0]))
    return pcd 


def define_plane(center):
    point1 = np.array([0, 1, center[2]])
    point2 = np.array([1, 2, center[2]])

    vector1 = point1 - center
    vector2 = point2 - center

    vector1 = vector1 / np.sqrt((vector1**2).sum())
    vector2 = vector2 / np.sqrt((vector2**2).sum())

    normal_plane = np.cross(vector1, vector2)
    return normal_plane / np.sqrt((normal_plane**2).sum())


def define_rotation_matrix(rot_axis, rot_angle):
    ux = rot_axis[:,0]
    uy = rot_axis[:,1]
    uz = rot_axis[:,2]

    R = np.repeat(np.eye(3)[None],len(rot_axis),0)
    R[:,0,0] = np.cos(rot_angle) + (ux**2) * (1 - np.cos(rot_angle))
    R[:,0,1] = ux * uy * (1-np.cos(rot_angle)) - uz * np.sin(rot_angle)
    R[:,0,2] = ux * uz * (1-np.cos(rot_angle)) + uy * np.sin(rot_angle)

    R[:,1,0] = uy * ux * (1-np.cos(rot_angle)) + uz * np.sin(rot_angle)
    R[:,1,1] = np.cos(rot_angle) + (uy**2)*(1-np.cos(rot_angle))
    R[:,1,2] = uy * uz * (1-np.cos(rot_angle)) - ux * np.sin(rot_angle)

    R[:,2,0] = ux * uz * (1-np.cos(rot_angle)) - uy * np.sin(rot_angle)
    R[:,2,1] = uy * uz * (1-np.cos(rot_angle)) + ux * np.sin(rot_angle)
    R[:,2,2] = np.cos(rot_angle) + (uz**2)*(1-np.cos(rot_angle))

    return R



def create_line_set(origin, target_points, color, multiple_origins=False):
    points = np.concatenate((origin, target_points), 0)
    step = len(target_points) if multiple_origins else 1
    u = 1 if multiple_origins else 0
    lines = [[idp*u, idp+step] for idp in range(len(target_points))]
    colors = [color for k in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def cast_rays(obstacle_mesh_list, origin, norm_dirs, multiple_origins=False):
    scene = o3d.t.geometry.RaycastingScene()
    obstacles = [
        o3d.t.geometry.TriangleMesh.from_legacy(mesh) for mesh in obstacle_mesh_list
    ]
    obstacles_id = [scene.add_triangles(ob) for ob in obstacles]
    u = 1 if multiple_origins else 0
    ray_list = [list(origin[jd*u]) + list(norm_dirs[jd]) for jd in range(len(norm_dirs))]
    rays = o3d.core.Tensor(ray_list, dtype=o3d.core.Dtype.Float32)
    results = scene.cast_rays(rays)
    return results


def mask_rays(far_target_points, rays_results):
    inter_dists = np.array([
        rays_results["t_hit"][ju].item() for ju in range(len(far_target_points))
    ])
    clean_mask = inter_dists == np.inf
    return clean_mask


def read_params(params):
    return params[0], params[1]

def define_obj_mesh_and_height(path, obj_name, obj_trans, obj_orient):
    obj_mesh = read_o3d_mesh(os.path.join(path, obj_name + ".ply"))
    obj_verts = np.array(obj_mesh.vertices)
    obj_faces = np.array(obj_mesh.triangles)
    obj_verts = obj_verts @ obj_orient.T + obj_trans
    obj_center = (obj_verts.max(0) + obj_verts.min(0)) / 2
    obj_dims = obj_verts.max(0) - obj_verts.min(0)
    obj_height = obj_dims[2]
    obj_mesh = create_o3d_mesh(obj_verts, obj_faces, [1,0,0])
    return obj_mesh, obj_height, obj_center


def respecify_target_points(target_points, origin, factor, projection=False):
    dirs = target_points - origin
    if projection:
        normal_plane = define_plane(origin)
        dirs = dirs - (np.dot(dirs, normal_plane)[..., None] * normal_plane) / np.sqrt((normal_plane**2).sum())
        # In case on of the direction is vertical, its projection to the plane that
        # we define with the normal is going to be the xero vector. So in for these cases
        # we exclude these vectors
        mask = ((dirs == np.zeros(3)).sum(1) != 3)
        dirs = dirs[mask]

    norm_dirs = dirs / (np.sqrt((dirs**2).sum(-1))[..., None])
    far_target_points = origin + factor * norm_dirs
    return far_target_points, norm_dirs

def specify_rotation_matrix(reference_vector, new_directions):
    rot_angles = np.arccos(reference_vector @ new_directions.T)

    ang_indexes = range(0, len(new_directions))
    ind = np.random.choice(ang_indexes, size=len(reference_vector))

    rot_angle = rot_angles[np.eye(len(new_directions))[ind].astype(bool)]

    print("Goya custom 4")

    R = np.repeat(np.eye(3)[None], len(reference_vector),  axis=0)
    R[:, 0, 0] = np.cos(-rot_angle)
    R[:, 0, 1] = - np.sin(-rot_angle)
    R[:, 1, 0] = np.sin(-rot_angle)
    R[:, 1, 1] = np.cos(-rot_angle)
    return R


def find_new_transl_and_gorient(R, body_params, obj_transl, pelvis, B, device):
    gorient = body_params["global_orient"]
    gor = aa2rotmat(gorient.unsqueeze(0)).reshape(-1, 3,3).cpu().numpy()
    new_gor = torch.tensor(R @ gor).to(device)
    new_gorient = rotmat2aa(new_gor).reshape(B, 3).to(torch.float32).to(device)
    new_transl = ((body_params["transl"].cpu().numpy() + pelvis - obj_transl)[:, None, :] @ R.transpose(0, 2, 1)).squeeze() + obj_transl - pelvis
    #new_transl = ((body_params["transl"].cpu().numpy() + pelvis - obj_transl)[:, None, :] @ R.transpose(0, 2, 1)).squeeze() + obj_transl 
    #new_transl = ((body_params["transl"].cpu().numpy() + pelvis)[:, None, :] @ R.transpose(0, 2, 1)).squeeze() - pelvis
    new_transl = torch.tensor(new_transl).to(torch.float32).to(device)
    return new_gorient, new_transl

def find_new_transl_and_gorient_test(R, body_params, body_joints, bjoints, B, device):
    # This function is for the case where we want to find the translation
    # and orientation of the body after rotating the body around a 
    # particular joint
    gorient = body_params["global_orient"]
    gor = aa2rotmat(gorient.unsqueeze(0)).reshape(-1, 3,3).cpu().numpy()
    new_gor = torch.tensor(R @ gor).to(device)
    new_gorient = rotmat2aa(new_gor).reshape(B, 3).to(torch.float32).to(device)
    new_transl = bjoints[:,0,:] - (body_joints[:,0,:] - body_params["transl"].cpu().numpy())
    new_transl = torch.tensor(new_transl).to(torch.float32).to(device)
    return new_gorient, new_transl


def reshaping(body_dict, device):
    body_dict.pop("fullpose_rotmat")
    for k,v in body_dict.items():
        if k != "transl" and  k != "betas":
            B = v.shape[0]
            v = rotmat2aa(v.cpu()).reshape(B,-1)
            body_dict[k] = v 
        body_dict[k] = v.to(device)
    return body_dict


def merge_dicts(list_body_dict):
    final_dict = {}
    for key, value in list_body_dict[0].items():
        final_dict[key] = torch.cat([
            d[key] for d in list_body_dict
        ], 0)
    return final_dict


def rotmat2aa(rotmat):
    '''
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = rotmat.shape[0]
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0,1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose


def aa2rotmat(axis_angle):
    '''
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nx1xnum_jointsx9
    '''
    batch_size = axis_angle.shape[0]
    pose_body_matrot = angle_axis_to_rotation_matrix(axis_angle.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    return pose_body_matrot


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4





def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def body_dict(X,index):
     body_dict = {"body_pose":aa2rotmat(torch.tensor(X["fullpose"][index][3:66]).reshape(1,21,3)).reshape(1,21,3,3).to(torch.float32).to("cuda"),
                  "transl":torch.tensor(X["transl"][index]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "global_orient":aa2rotmat(torch.tensor(X["fullpose"][index][0:3]).reshape(1,3)).reshape(1,3,3).to(torch.float32).to("cuda"),
                  "jaw_pose":torch.tensor(X["fullpose_rotmat"][index,22,:,:]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "leye_pose":torch.tensor(X["fullpose_rotmat"][index,23,:,:]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "reye_pose":torch.tensor(X["fullpose_rotmat"][index,24,:,:]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "left_hand__pose":torch.tensor(X["fullpose_rotmat"][index,25:40,:,:]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "right_hand_pose":torch.tensor(X["fullpose_rotmat"][index,40:55,:,:]).unsqueeze(0).to(torch.float32).to("cuda"),
                  "betas":torch.tensor(X["betas"][index]).unsqueeze(0).to(torch.float32).to("cuda")
      }
     return body_dict
 
def vis_bodies(indexes, male_model, female_model,mfaces,ffaces, gender):
    for i in range(len(indexes)):
        if gender[i] == 1:
              bm1 = male_model(**body_dict(X, indexes[i]))
        else:
              bm1 = female_model(**body_dict(X, indexes[i]))
        verts = bm1.vertices.detach().squeeze().cpu().numpy()
        mesh_box = create_o3d_box_mesh_vertical_y(verts)
        if gender[i] == 1:
          mesh = create_o3d_mesh(verts,mfaces,[0.3,0.2,0.6])
        else:
          mesh = create_o3d_mesh(verts,ffaces,[0.4,0.2,0.7])
        o3d.visualization.draw_geometries([mesh,mesh_box])


