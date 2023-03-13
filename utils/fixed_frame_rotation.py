import numpy as np

def rotation_matrix_3d(angle_x, angle_y, angle_z):
    # Convert angles to radians
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    # Compute sine and cosine of the angles
    sx, cx = np.sin(angle_x), np.cos(angle_x)
    sy, cy = np.sin(angle_y), np.cos(angle_y)
    sz, cz = np.sin(angle_z), np.cos(angle_z)

    # Compute the rotation matrix
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)

    return R


def rotation_matrix_to_quaternion(R):
    # Compute the trace of the rotation matrix
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

if __name__ == '__main__':
    Rx = 25
    Ry = 0
    Rz = 0
    R_world_camera = np.asarray([[-1, 0, 0], 
                                 [0, 1, 0], 
                                 [0, 0, -1]])
    
    R_camera_new_camera = rotation_matrix_3d(angle_x=Rx, angle_y=Ry, angle_z=Rz)

    # R_world_new_camera = R_camera_new_camera.dot(R_world_camera)
    R_world_new_camera = R_world_camera.dot(R_camera_new_camera)
    print(R_world_new_camera)

    quat_world_new_camera = rotation_matrix_to_quaternion(R_world_new_camera)
    print(quat_world_new_camera)