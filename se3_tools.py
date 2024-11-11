import numpy as np

from numba import njit

from scipy.spatial.transform import Rotation


@njit('float64[:, :](float64[:, :], float64)', cache=True)
def _rotValidate(rot, eps=1e-5):
    """
    rotValidate raises an exception if the rotation matrix is not orthonormal

    Args:
        rot: A 3x3 matrix

    Returns:
        R: If C is a valid rotation matrix, it is returned. Otherwise the program raises an exception.
    """

    assert rot.shape == (3, 3)

    RtR = np.asfortranarray(rot.T) @ np.ascontiguousarray(rot)

    E = RtR - np.eye(3)
    err = np.max(np.abs(E))

    if err > eps:
        raise Exception('The rotation matrix is not valid.')

    return rot


def rotValidate(rot, eps=1e-5):
    _rotValidate(rot, eps)


@njit('float64[:, :](float64[:, :], float64)', cache=True)
def _poseValidate(pose, eps=1e-5):
    """
    tranValidate causes an exception if the transformation matrix is not in SE(3)

    Args:
        pose: A 4x4 matrix

    Returns:
        T: If T is a valid transformation matrix, it is returned. Otherwise program raises an exception.
    """

    assert pose.shape == (4, 4)
    _rotValidate(pose[:3, :3], eps)

    err = np.max(np.abs(np.array([0., 0., 0., 1.]) - pose[3]))

    if err > eps:
        raise Exception('The bottom row of the transformation matrix is not [0, 0, 0, 1]')

    return pose


def poseValidate(pose, eps=1e-5):
    _poseValidate(pose, eps)


@njit('float64[:, :](float64[:, :])')
def change_of_twist_frame_matrix(pose):
    """
    Function used to get matrix to convert a twist from one frame of reference to another.

    Let V_EE be the twist in the end-effector frame and V_B be the twist in the base frame. Then V_B = T @ T_EE
    where T = change_of_twist_frame_matrix(eef_pose_in_base)
    """

    T = np.zeros((6, 6))
    T[:3, :3] = T[3:6, 3:6] = pose[:3, :3]

    return T


@njit('float64[:, :](float64[:])', cache=True)
def skew_symmetric_matrix(vec):
    """
    hat builds the 3x3 skew symmetric matrix from the 3x1 input or 3x3 from 6x1 input

    Args:
        vec: 3 vector phi or 6x1 vector xi = [rho, phi]

    Returns:
        vechat: the 3x3 skew symmetric matrix that can be used to implement the cross product, or 4x4 transformation
                matrix

    """

    if len(vec) == 3:

        vechat = np.array([[0., -vec[2], vec[1]],
                           [vec[2], 0., -vec[0]],
                           [-vec[1], vec[0], 0.]])

    elif len(vec) == 6:

        vechat = np.zeros((4, 4))
        vechat[:3, :3] = skew_symmetric_matrix(vec[3:])
        vechat[:3, 3] = vec[:3]

    else:
        raise Exception('Invalid vector length for hat operator\n')

    return vechat


@njit('float64[:, :](float64[:])', cache=True)
def _curlyhat(vec):
    """
    curlyhat builds the 6x6 curly hat matrix from the 6x1 input

    Args:
        vec: 6x1 vector xi

    Returns:
        veccurlyhat: the 6x6 curly hat matrix

    """

    assert len(vec) == 6

    phihat = skew_symmetric_matrix(vec[3:])
    veccurlyhat = np.zeros((6, 6))
    veccurlyhat[:3, :3] = phihat
    veccurlyhat[3:, 3:] = phihat
    veccurlyhat[:3, 3:] = skew_symmetric_matrix(vec[:3])

    return veccurlyhat


@njit('float64[:, :](float64[:, :])', cache=True)
def poseAd(T):
    """
    TranAd builds the 6x6 transformation matrix from the 4x4 one

    Args:
        T: 4x4 transformation matrix

    Returns:
        AdT: 6x6 transformation matrix

    """

    _poseValidate(T, 1e-5)

    R = T[:3, :3]
    t = T[:3, 3]

    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:6, 3:6] = R
    AdT[:3, 3:] = np.asfortranarray(skew_symmetric_matrix(t)) @ np.ascontiguousarray(R)

    return AdT


@njit('float64[:, :](float64[:, :])', cache=True)
def sqrtm(A):
    """
    Computes matrix square root using an iterative method. Code adapted from
    https://github.com/XD-DENG/sqrt-matrix/blob/master/python/Babylonian_method.py

    Args:
        A: Input matrix

    Returns:
        X: Matrix square root of A such that A = X^2
    """
    error_tolerance = 1.5e-8

    X = np.eye(len(A))
    done = False

    while not done:
        X_old = X
        X = (X + np.asfortranarray(A) @ np.ascontiguousarray(np.linalg.inv(X))) / 2

        # detect the maximum value in the error matrix
        error = np.max(np.abs(X - X_old))

        if error < error_tolerance:
            done = True
    return X


@njit('float64[:, :](float64[:], int64)', cache=True)
def _vec2rotSeries(phi, N):
    """
    Build a rotation matrix using the exponential map series with N elements in the series

    Args:
        phi: 3x1 vector of angles
        N: Number of terms to include in the series

    Returns:
        R: 3x3 rotation matrix

    """

    assert len(phi) == 3

    R = np.eye(3)
    xM = np.eye(3)
    cmPhi = skew_symmetric_matrix(phi)
    for n in range(1, N + 1):
        xM = xM @ (cmPhi / n)
        R = R + xM

    # Project the resulting rotation matrix back onto SO(3)
    R = R @ np.linalg.inv(sqrtm(R.T @ R))

    return R


@njit('float64[:, :](float64[:])', cache=True)
def _vec2Q(vec):
    """
    Construction of the 3x3 Q matrix.

    Args:
        vec: a 6x1 vector

    Returns:
        Q: the 3x3 Q matrix.

    """

    vec = np.ascontiguousarray(vec).reshape(-1, 1)

    rho = vec[:3]
    phi = vec[3:]

    ph = np.linalg.norm(phi)
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    rx = skew_symmetric_matrix(rho.reshape(-1))
    px = skew_symmetric_matrix(phi.reshape(-1))

    t1 = 0.5 * rx
    t2 = ((ph - sph) / ph3) * (
            np.asfortranarray(px) @ np.ascontiguousarray(rx) + np.asfortranarray(rx) @ np.ascontiguousarray(
        px) + np.asfortranarray(px) @ np.asfortranarray(rx) @ np.ascontiguousarray(px))
    m3 = (1 - 0.5 * ph2 - cph) / ph4
    t3 = -m3 * (np.asfortranarray(px) @ np.asfortranarray(px) @ np.ascontiguousarray(rx) + np.asfortranarray(
        rx) @ np.asfortranarray(px) @ np.ascontiguousarray(px) - 3 * np.asfortranarray(px) @ np.asfortranarray(
        rx) @ np.ascontiguousarray(px))
    m4 = 0.5 * (m3 - 3 * (ph - sph - ph3 / 6) / ph5)
    t4 = -m4 * (np.asfortranarray(px) @ np.asfortranarray(rx) @ np.asfortranarray(px) @ np.ascontiguousarray(
        px) + np.asfortranarray(px) @ np.asfortranarray(px) @ np.asfortranarray(rx) @ np.ascontiguousarray(px))

    Q = t1 + t2 + t3 + t4

    return Q


@njit('float64[:, :](float64[:], int64)', cache=True)
def _vec2jacSeries(vec, N):
    """
    Construction of the J matrix from Taylor series

    Args:
        vec: a 3x1 or 6x1 vector
        N: The number of terms to include in the series

    Returns:
        J: the 3x3 J matrix

    """

    if vec.shape[0] == 3:

        J = np.eye(3)
        pxn = np.eye(3)
        px = skew_symmetric_matrix(vec)

        for n in range(1, N + 1):
            pxn = np.asfortranarray(pxn) @ np.ascontiguousarray(px) / (n + 1)
            J = J + pxn

    elif vec.shape[0] == 6:

        J = np.eye(6)
        pxn = np.eye(6)
        px = _curlyhat(vec)

        for n in range(1, N + 1):
            pxn = np.asfortranarray(pxn) @ np.ascontiguousarray(px) / (n + 1)
            J = J + pxn

    else:
        raise Exception('Invalid input vector length')

    return J


@njit('float64[:, :](float64[:])', cache=True)
def _vec2jac(vec):
    """
    vec2jac construction of the 3x3 J matrix or the 6x6 J matrix

    Args:
        vec: a 3x1 vector or 6x1 vector xi

    Returns:
        J: the 3x3 Jacobin matrix or 6x6 Jacobin matrix, depending on the input
    """

    tolerance = 1e-12

    if len(vec) == 3:
        phi = vec

        ph = np.linalg.norm(phi)
        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            J = _vec2jacSeries(phi, 10)
        else:
            axis = (phi / ph).reshape(-1, 1)

            cph = (1 - np.cos(ph)) / ph
            sph = np.sin(ph) / ph

            J = sph * np.eye(3) + (1 - sph) * axis @ axis.T + cph * skew_symmetric_matrix(axis.reshape(-1))

    elif len(vec) == 6:

        phi = vec[3:]

        ph = np.linalg.norm(phi)
        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            J = _vec2jacSeries(vec, 10)
        else:
            Jsmall = _vec2jac(phi)
            Q = _vec2Q(vec)
            J = np.zeros((6, 6))
            J[:3, :3] = Jsmall
            J[3:, 3:] = Jsmall
            J[:3, 3:] = Q
    else:
        raise Exception('Invalid input length.')

    return J


@njit('int64(int64)', cache=True)
def factorial(n):
    if n < 1:
        return 1
    else:
        return n * factorial(n - 1)


@njit('int64(int64, int64)', cache=True)
def combination(m, k):
    if k <= m:
        return factorial(m) / (factorial(k) * factorial(m - k))
    else:
        return 0


@njit('float64(int64)', cache=True)
def bernoullinumber(m):
    """
    Generate the kth bernoulli number

    Args:
        k: int that specifies which bernoulli number to generate

    Returns:
        b: kth bernoulli number

    """
    if m == 0:
        return 1.

    elif m == 1:
        return 1 / 2

    elif m % 2 != 0:
        return 0.

    else:
        t = 0.
        for k in range(0, m):
            t += combination(m, k) * bernoullinumber(k) / (m - k + 1)
        return 1. - t


@njit('float64[:, :](float64[:], int64)', cache=True)
def _vec2jacInvSeries(vec, N):
    """
    Construction of the 3x3 J^-1 matrix or 6x6 J^-1 matrix. Series representation

    Args:
        vec: 3x1 vector or 6x1 vector

    Returns:
        invJ: 3x3 inv(J) matrix or 6x6 inv(J) matrix
    """

    if len(vec) == 3:

        invJ = np.eye(3)
        pxn = np.eye(3)
        px = skew_symmetric_matrix(vec)

        for n in range(1, N + 1):
            pxn = np.asfortranarray(pxn) @ np.ascontiguousarray(px) / n
            invJ = invJ + bernoullinumber(n) * pxn

    elif len(vec) == 6:

        invJ = np.eye(6)
        pxn = np.eye(6)
        px = _curlyhat(vec)

        for n in range(1, N + 1):
            pxn = np.asfortranarray(pxn) @ np.ascontiguousarray(px) / n
            invJ = invJ + bernoullinumber(n) * pxn

    else:
        raise Exception('Invalid input vector length\n')

    return invJ


@njit('float64(float64)', cache=True)
def cot(x):
    return np.cos(x) / np.sin(x)


@njit('float64[:, :](float64[:])', cache=True)
def _vec2jacInv(vec):
    """
    Construction of the 3x3 J^-1 matrix or 6x6 J^-1 matrix in closed form

    Args:
        vec: 3x1 vector or 6x1 vector

    Returns:
        invJ: 3x3 inv(J) matrix or 6x6 inv(J) matrix

    """

    tolerance = 1e-12

    if len(vec) == 3:

        phi = vec

        ph = np.linalg.norm(phi)
        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            invJ = _vec2jacInvSeries(phi, 10)

        else:
            axis = (phi / ph).reshape(-1, 1)
            ph_2 = 0.5 * ph

            invJ = ph_2 * cot(ph_2) * np.eye(3) + (1 - ph_2 * cot(ph_2)) * axis @ axis.T - ph_2 * skew_symmetric_matrix(
                axis.reshape(-1))

    elif len(vec) == 6:

        phi = vec[3:]
        ph = np.linalg.norm(phi)

        if ph < tolerance:
            # If the angle is small, fall back on the series representation
            invJ = _vec2jacInvSeries(vec, 10)

        else:
            invJsmall = _vec2jacInv(phi)
            Q = _vec2Q(vec)
            invJ = np.zeros((6, 6))
            invJ[:3, :3] = invJsmall
            invJ[:3, 3:] = - np.asfortranarray(invJsmall) @ np.asfortranarray(Q) @ np.ascontiguousarray(invJsmall)
            invJ[3:, 3:] = invJsmall

    return invJ


@njit('float64[:, :](float64[:])', cache=True)
def so3_exp(phi):
    """
    Build a rotation matrix using the exponential map

    Args:
        phi: 3x1 vector

    Returns:
        R: 3x3 rotation matrix

    """

    assert len(phi) == 3

    angle = np.linalg.norm(phi)
    tolerance = 1e-12

    # Check for a small angle
    if angle < tolerance:
        # If the angle is small, fall back on the series representation
        R = _vec2rotSeries(phi, 10)

    else:
        axis = (phi / angle).reshape(-1, 1)

        cp = np.cos(angle)
        sp = np.sin(angle)

        R = cp * np.eye(3) + (1 - cp) * axis @ axis.T + sp * skew_symmetric_matrix(axis.reshape(-1))

    _rotValidate(R, 1e-5)

    return R


@njit('float64[:, :](float64[:])', cache=True)
def se3_exp(vec):
    """
    Build a transformation matrix using the exponential map, closed form

    Args:
        vec: 6x1 vector

    Returns:
        T: 4x4 transformation matrix

    """
    assert len(vec) == 6

    phi = vec[3:]
    rho = vec[:3]

    R = so3_exp(phi)
    J = _vec2jac(phi)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asfortranarray(J) @ np.ascontiguousarray(rho)

    _poseValidate(T, 1e-5)

    return T


@njit('float64[:](float64[:, :])', cache=True)
def so3_log(R):
    """
    compute the matrix log of the rotation matrix R.

    Args:
        R: a 3x3 rotation matrix

    Returns:
        phi: a 3x1 vector (axis * angle) computed from R.
    """

    _rotValidate(R, 1e-5)

    # Rotation matrices have complex eigenvalues. We need to change R to be complex to be able to use numba and
    # np.linalg.eig()
    R = R + 0j
    w, v = np.linalg.eig(R)
    R = np.real(R)  # After eigen decomposition change R back to a real matrix.
    for i in range(3):
        if abs(np.real(w[i]) - 1) < 1e-10:
            # check if one remaining eigenvalue is the complex conjugate of the other
            idx = [0, 1, 2]
            idx.remove(i)
            dif = w[idx[0]] - np.conj(w[idx[1]])
            if abs(np.real(dif)) > 1e-10 or abs(np.imag(dif)) > 1e-10:
                continue

            a = np.real(v[:, i:i + 1])
            a = (a / np.sqrt(np.asfortranarray(a.T) @ np.ascontiguousarray(a))).reshape(-1)
            traceR = np.trace(R)
            if traceR > 3:
                if traceR - 3 < 1e-10:
                    traceR = 3
                else:
                    raise Exception('The trace of the input rotation matrix is > 3.')

            phim = np.arccos((traceR - 1) / 2)
            phi = phim * a
            if np.abs(R - so3_exp(phi)).max() > np.abs(R - so3_exp(-phi)).max():
                phi = -phi
            else:
                continue

            break

    return phi


@njit('float64[:, :](float64[:])', cache=True)
def rotvec2rot(rotvec):
    return so3_exp(rotvec)


@njit('float64[:](float64[:, :])', cache=True)
def rot2rotvec(rot):
    return so3_log(rot)


@njit('float64[:](float64[:, :])', cache=True)
def rot2axis_angle(rot):
    rotvec = rot2rotvec(rot)
    angle = np.linalg.norm(rotvec)
    axis = rotvec / angle
    return np.concatenate((axis, np.array([angle])))


@njit('float64[:, :](float64[:])', cache=True)
def axis_angle2rot(axis_angle):
    rotvec = axis_angle[:3] * axis_angle[3]
    rot = rotvec2rot(rotvec)
    return rot


@njit('float64[:](float64[: , :])', cache=True)
def se3_log(T):
    """
    Compute the matrix log of the transformation matrix T.

    Args:
        T: a 4x4 transformation matrix

    Returns:
        p: a 6x1 vector in tangent coordinates computed from T
    """
    # tranValidate(T)

    R = T[:3, :3]
    r = T[:3, 3]

    phi = so3_log(R)
    invJ = _vec2jacInv(phi)

    rho = np.asfortranarray(invJ) @ np.ascontiguousarray(r)
    p = np.concatenate((rho, phi), axis=0)
    return p


@njit('float64[:, :](float64[:])', cache=True)
def quat2rot(quaternion):
    """

    :param quaternion: (qi, qj, qk, qr) where qr is the scalar part of the quaternion and (qi, qj, qk,) is the vector
                        part.
    :return: R: A rotation matrix of shape (3, 3).
    """
    quaternion = quaternion / np.linalg.norm(quaternion)

    qi, qj, qk, qr = quaternion

    R = np.array([[1 - 2 * (qj ** 2 + qk ** 2), 2 * (qi * qj - qk * qr), 2 * (qi * qk + qj * qr)],
                  [2 * (qi * qj + qk * qr), 1 - 2 * (qi ** 2 + qk ** 2), 2 * (qj * qk - qi * qr)],
                  [2 * (qi * qk - qj * qr), 2 * (qj * qk + qi * qr), 1 - 2 * (qi ** 2 + qj ** 2)]],
                 dtype=np.float64)

    _rotValidate(R, 1e-5)

    return R


@njit('float64[:](float64[:, :])', cache=True)
def rot2quat(R):
    _rotValidate(R, 1e-3)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2][1] - R[1][2]) * s
        qy = (R[0][2] - R[2][0]) * s
        qz = (R[1][0] - R[0][1]) * s

    else:
        if ((R[0][0] > R[1][1]) and (R[0][0] > R[2][2])):
            s = 2.0 * np.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
            qw = (R[2][1] - R[1][2]) / s
            qx = 0.25 * s
            qy = (R[0][1] + R[1][0]) / s
            qz = (R[0][2] + R[2][0]) / s
        elif (R[1][1] > R[2][2]):
            s = 2.0 * np.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
            qw = (R[0][2] - R[2][0]) / s
            qx = (R[0][1] + R[1][0]) / s
            qy = 0.25 * s
            qz = (R[1][2] + R[2][1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
            qw = (R[1][0] - R[0][1]) / s
            qx = (R[0][2] + R[2][0]) / s
            qy = (R[1][2] + R[2][1]) / s
            qz = 0.25 * s

    return np.array([qx, qy, qz, qw])


@njit('float64[:, :](float64[:])', cache=True)
def posevec2inv_pose(posevec):
    """
    Function used to convert a posevec to an extrinsic matrix.

    Args:
        posevec: Position concatenated with a quaternion.

    Returns:
        T_OW: Extrinsic Matrix. This is a transformation that can be used to map a point in the world frame to the
              object frame.

    """
    R = quat2rot(posevec[3:])

    # {O} = Object frame and {W} = World frame
    T_OW = np.eye(4)
    T_OW[:3, :3] = R.T
    T_OW[:3, 3] = - R.T @ np.ascontiguousarray(posevec[:3])

    return T_OW


@njit('float64[:, :](float64[:, :])', cache=True)
def pose_inv(pose):
    """
    Function used to convert a pose to an extrinsic matrix.

    Args:
        posevec: Position concatenated with a quaternion.

    Returns:
        T_OW: Extrinsic Matrix. This is a transformation that can be used to map a point in the world frame to the
              object frame.

    """
    R = pose[:3, :3]

    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])

    return T


@njit('float64[:, :](float64[:])', cache=True)
def posevec2pose(posevec):
    T_WO = np.eye(4)
    T_WO[:3, :3] = quat2rot(posevec[3:])
    T_WO[:3, 3] = posevec[:3]

    return T_WO


@njit('float64[:](float64[:, :])', cache=True)
def pose2posevec(pose):
    quaternion = rot2quat(pose[:3, :3])
    position = pose[:3, 3]
    return np.concatenate((position, quaternion), axis=0)


@njit('float64[:](float64[:])', cache=True)
def posevec_inv(posevec):
    pose = posevec2pose(posevec)
    _pose_inv = pose_inv(pose)
    return pose2posevec(_pose_inv)


@njit('float64[:](float64[:], float64[:, :])', cache=True)
def twist_in_A_to_twist_in_B(twist_in_A, pose_A_in_B):
    Adjoint = poseAd(pose_A_in_B)
    twist_in_B = np.ascontiguousarray(Adjoint) @ np.ascontiguousarray(twist_in_A)
    return twist_in_B


@njit('float64[:](float64[:], float64[:, :])', cache=True)
def wrench_in_A_to_wrench_in_B(wrench_in_A, pose_A_in_B):
    Adjoint = poseAd(pose_A_in_B)
    wrench_in_B = np.ascontiguousarray(Adjoint) @ np.ascontiguousarray(wrench_in_A)
    return wrench_in_B


def euler2rot(seq, angles, degrees=False):
    return Rotation.from_euler(seq, angles, degrees).as_matrix()


def rot2euler(seq, rot, degrees=False):
    return Rotation.from_matrix(rot).as_euler(seq, degrees=degrees)


def quat2euler(seq, quat, degrees=False):
    return Rotation.from_quat(quat).as_euler(seq, degrees=degrees)


def euler2quat(seq, angles, degrees=False):
    return Rotation.from_euler(seq, angles, degrees).as_quat()


def euler2rotvec(seq, angles, degrees=False):
    return Rotation.from_euler(seq, angles, degrees).as_rotvec()


def rotvec2euler(seq, rotvec, degrees=False):
    return Rotation.from_rotvec(rotvec).as_euler(seq, degrees)


def axis_angle2euler(seq, axis_angle, degrees=False):
    return rotvec2euler(seq, axis_angle[0] * axis_angle[1:], degrees)


def euler2axis_angle(seq, angles, degrees=False):
    rotvec = euler2rotvec(seq, angles, degrees)
    angle = np.linalg.norm(rotvec)
    axis = rotvec / angle
    return np.array([angle, axis[0], axis[1], axis[2]])


@njit('float64[:, :](float64[:], float64[:, :])', cache=True)
def make_pose(translation, rot):
    """
    Makes a homogenous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation: a 3-dim iterable
        rotation: a 3x3 matrix

    Returns:
        pose: a 4x4 homogenous matrix
    """
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = translation
    return pose


def roll_pitch_yaw_to_mat(r, p, y):
    return Rotation.from_euler('XYZ', np.array([r, p, y])).as_matrix()


@njit('float64[:](float64[:], float64[:, :])', cache=True)
def apply_tansformation_matrix(vector, matrix):
    vector = np.concatenate((vector, np.array([1.])))
    vector = np.ascontiguousarray(matrix) @ np.ascontiguousarray(vector)
    vector[:3] /= vector[3]
    vector = vector[:3]
    return vector


def quat_represetion(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


@njit('float64[:](float64[:], float64[:])', cache=True)
def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,), dtype=np.float64)


@njit('float64[:](float64[:])', cache=True)
def quat_conjugate(quaternion):
    """Return conjugate of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True
    """
    return np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)


@njit('float64[:](float64[:])', cache=True)
def quat_inverse(quaternion):
    """Return inverse of quaternion.
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True
    """
    return quat_conjugate(quaternion) / (np.ascontiguousarray(quaternion) @ np.ascontiguousarray(quaternion))


@njit('float64[:](float64[:])', cache=True)
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """

    eps = np.finfo(float).eps * 4

    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < eps:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = np.arccos(d) + spin * np.pi
    if abs(angle) < eps:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quat(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2), dtype=np.float64)
