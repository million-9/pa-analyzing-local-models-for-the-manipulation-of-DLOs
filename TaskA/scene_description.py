import Sofa
import Sofa.Core

import numpy as np
import matplotlib.pyplot as plt
# from Planner0813 import Planner0813

from functools import partial
from typing import Optional, Tuple, Dict
from pathlib import Path

from sofa_env.sofa_templates.collision import add_collision_model, CollisionModelType, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding, RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.rope import RopeCollisionType, Rope, poses_for_circular_rope, poses_for_linear_rope, ROPE_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST

from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.rope_threading.sofa_objects.transfer_rope import TransferRope, TRANSFER_ROPE_PLUGIN_LIST
from sofa_env.utils.camera import determine_look_at
from sofa_env.utils.math_helper import euler_to_rotation_matrix, rotation_matrix_to_quaternion, multiply_quaternions,quaternion_to_euler_angles
from sofa_env.utils.pivot_transform import sofa_orientation_to_camera_orientation

PLUGIN_LIST = (
    [
        "SofaPython3",
        "SofaMiscCollision",
        "Sofa.Component.MechanicalLoad",  # [PlaneForceField]
        "Sofa.Component.Topology.Container.Grid",  # [CylinderGridTopology]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + TRANSFER_ROPE_PLUGIN_LIST
)

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"

ROPE_POSITIONS = None
PROGRAM_STEP = 0
ROPE_POSITIONS_HISTORY = np.zeros((1600, 100, 3))
STATE = 0
CURRENT_STEP = None
CURRENT_POINT = None
X_GRIPPER = None
X_GRIPPER_LAST = None
V_GRIPPER = None
TRJ_GRIPPER = None
TRJ_IDX = None

X_MID_GRIPPER = None
X_MID_GRIPPER_LAST = None
V_MID_GRIPPER = None
PAUSE_COUNT = None

GRIP_POINT = None
MID_GRIPPER_POSE_AFTER_GRASP = None
MID_GRIPPER_POSE_TO_DRIVE = None
YAW_OUT = None
AX = None
PATH_CONCAT = None

def pause(steps=50):
    global PAUSE_COUNT
    if PAUSE_COUNT is None:
        PAUSE_COUNT = 1
    else:
        PAUSE_COUNT += 1

    if PAUSE_COUNT >= steps:
        PAUSE_COUNT = None
        return 1
    else:
        return 0


def release_object(gripper):
    theta_des = 15
    d_theta_limit = 1

    theta_current = gripper.get_angle()
    d_theta = max(min(theta_des - theta_current, d_theta_limit), -d_theta_limit)
    theta_current += d_theta
    gripper.set_angle(theta_current)
    if np.abs(theta_des - theta_current) < 1:
        return 1
    return 0


def regrasp_object(gripper):
    theta_desired = 2
    d_theta_limit_ = 1

    theta_cur = gripper.get_angle()
    d_theta_ = max(min(theta_desired - theta_cur, d_theta_limit_), -d_theta_limit_)
    theta_cur += d_theta_
    gripper.set_angle(theta_cur)
    check = gripper.get_angle()
    if np.abs(theta_desired - theta_cur) < 1:
        return pause()
    return 0


def angle_transform(angle_in, forward=1):
    if forward:
        if angle_in < 0:
            angle_in = angle_in + 2*np.pi
        angle_in = angle_in - np.pi
        angle_in = max(min(angle_in, 0.95*np.pi), -0.95*np.pi)
        angle_out = np.tan(0.5 * angle_in)
    else:
        angle_in = 2 * np.arctan(angle_in)
        angle_out = angle_in + np.pi
    return angle_out


def gripper_2_task_angle(gripper_angle):
    if gripper_angle >= 0:
        task_angle = 2 * np.arctan(gripper_angle) - np.pi
    else:
        task_angle = 2 * np.arctan(gripper_angle) + np.pi
    return task_angle


def task_2_gripper_angle(task_angle):
    if task_angle < 0:
        gripper_angle = np.tan((task_angle + np.pi)/2)
    else:
        gripper_angle = np.tan((task_angle - np.pi)/2)
    return gripper_angle


def get_gripper_pose(gripper):
    return gripper.get_pose().copy()[:3], gripper.get_pose().copy()[4]


def compute_trajectory(v_gripper, a_gripper, b1, yaw_des_task, use_mid_gripper=False, debug=False):
    global X_GRIPPER, X_MID_GRIPPER
    if use_mid_gripper:
        x_gripper = X_MID_GRIPPER.copy()
    else:
        x_gripper = X_GRIPPER.copy()
    yaw_0_task = gripper_2_task_angle(gripper_angle=x_gripper[4]) # angle_transform(angle_in=x_gripper[4], forward=0)

    b0 = x_gripper[:3]

    n = 100
    t = np.linspace(0, 1, n)
    dummy_cos = np.cos(yaw_0_task) + t * (np.cos(yaw_des_task) - np.cos(yaw_0_task))
    dummy_sin = np.sin(yaw_0_task) + t * (np.sin(yaw_des_task) - np.sin(yaw_0_task))
    dummy_angle = np.arctan2(dummy_sin, dummy_cos)
    if False:  # use_mid_gripper:
        plt.figure()
        plt.plot(dummy_angle)
        plt.plot(0, dummy_angle[0], 'r.')
        plt.plot(dummy_angle.size-1, dummy_angle[-1], 'r.')
        plt.draw()
        plt.pause(0.1)
        print('a')
    path_gripper = np.zeros((n, 5))
    path_gripper[:, 0] = b0[0] + t * (b1[0] - b0[0])
    path_gripper[:, 1] = b0[1] + t * (b1[1] - b0[1])
    path_gripper[:, 2] = b0[2] + t * (b1[2] - b0[2])
    path_gripper[:, 3] = dummy_angle # yaw0 + t * (yaw1 - yaw0)

    for i in range(1, n):
        path_gripper[i, 4] = path_gripper[i-1, 4] + np.linalg.norm(path_gripper[i, :3]-path_gripper[i-1, :3])

    # Compute time-parameterization
    qf = path_gripper[-1, 4]
    ta = min(v_gripper/a_gripper, np.sqrt(qf/a_gripper))
    tf = ta + qf/(a_gripper*ta)
    dt = 0.01
    n_size = int(np.ceil(tf / dt))

    no_position_delta = False
    # Case when delta position is zero
    if n_size == 1:
        no_position_delta = True
        n_size = 10

    trj_gripper = np.zeros((n_size, 4))
    t = np.arange(n_size) * dt
    q_t = 0.5 * a_gripper * ta ** 2 + v_gripper * (t - ta)
    q_t[t <= ta] = 0.5 * a_gripper * t[t <= ta] ** 2
    q_t[t >= tf-ta] = qf - 0.5 * a_gripper * (tf - t[t >= tf-ta]) ** 2

    for i in range(n_size):
        if no_position_delta:
            idx_inter = int(np.floor(i/(n_size-1) * path_gripper.shape[0]))
            idx_inter = min(max(idx_inter, 0), path_gripper.shape[0]-1)
            trj_gripper[i, :3] = path_gripper[0, :3]
            trj_gripper[i, 3] = path_gripper[idx_inter, 3]
        else:
            q_curr = q_t[i]
            idx0_, idx1_ = np.argsort(np.abs(q_curr - path_gripper[:, 4]))[:2]
            if idx0_ < idx1_:
                idx0 = idx0_
                idx1 = idx1_
            else:
                idx0 = idx1_
                idx1 = idx0_
            s_inter = (q_curr - path_gripper[idx0, 4]) / (path_gripper[idx1, 4] - path_gripper[idx0, 4])
            trj_gripper[i, :3] = path_gripper[idx0, :3] + s_inter * (path_gripper[idx1, :3] - path_gripper[idx0, :3])
            cos_0 = np.cos(path_gripper[idx0, 3])
            cos_1 = np.cos(path_gripper[idx1, 3])
            sin_0 = np.sin(path_gripper[idx0, 3])
            sin_1 = np.sin(path_gripper[idx1, 3])
            trj_gripper[i, 3] = np.arctan2(sin_0 + s_inter * (sin_1 - sin_0), cos_0 + s_inter * (cos_1 - cos_0))
            # trj_tip[i, :] = path_tip[idx0, :4] + s_inter * (path_tip[idx1, :4] - path_tip[idx0, :4])

    if False:
        plt.plot(trj_gripper[:, 3])
        plt.draw()
        plt.pause(0.1)
        print('a')
    if debug:
        print('a')

    global TRJ_GRIPPER
    TRJ_GRIPPER = trj_gripper

    # fig, ax = plt.subplots(2)
    # ax[0].plot(trj_gripper[:, 0])
    # ax[0].plot(trj_gripper[:, 1])
    # ax[0].plot(trj_gripper[:, 2])
    # ax[0].plot(trj_gripper[:, 3])
    # ax[1].plot(trj_gripper[:, 4])

    # plt.show()

    return 1


def compute_continuous_trajectory(v_gripper, a_gripper, path, use_mid_gripper=False, debug=False):
    global X_GRIPPER, X_MID_GRIPPER
    if use_mid_gripper:
        x_gripper = X_MID_GRIPPER.copy()
    else:
        x_gripper = X_GRIPPER.copy()
    yaw_0_task = gripper_2_task_angle(gripper_angle=x_gripper[4])

    p0 = np.asarray([x_gripper[0], x_gripper[1], x_gripper[2], yaw_0_task]).reshape(1, -1)
    path = np.concatenate([p0, path], axis=0)
    path_w_length = np.concatenate([path, np.zeros((path.shape[0], 1))], axis=1)

    for i in range(1, path_w_length.shape[0]):
        path_w_length[i, -1] = path_w_length[i-1, -1] + np.linalg.norm(path_w_length[i, :3] - path_w_length[i-1, :3])

    # Compute time-parameterization
    qf = path_w_length[-1, -1]
    ta = min(v_gripper/a_gripper, np.sqrt(qf/a_gripper))
    tf = ta + qf/(a_gripper*ta)
    dt = 0.01
    n_size = int(np.ceil(tf / dt))

    no_position_delta = False
    # Case when delta position is zero
    if n_size == 1:
        no_position_delta = True
        n_size = 10

    trj_gripper = np.zeros((n_size, 4))
    t = np.arange(n_size) * dt
    q_t = 0.5 * a_gripper * ta ** 2 + v_gripper * (t - ta)
    q_t[t <= ta] = 0.5 * a_gripper * t[t <= ta] ** 2
    q_t[t >= tf-ta] = qf - 0.5 * a_gripper * (tf - t[t >= tf-ta]) ** 2

    s_trj = []

    for i in range(n_size):
        if no_position_delta:
            idx_inter = int(np.floor(i/(n_size-1) * path_w_length.shape[0]))
            idx_inter = min(max(idx_inter, 0), path_w_length.shape[0]-1)
            trj_gripper[i, :3] = path_w_length[0, :3]
            trj_gripper[i, 3] = path_w_length[idx_inter, 3]
        else:
            q_curr = q_t[i]
            idx_ref = np.argmin(np.abs(q_curr - path_w_length[:, -1]))
            q_ref = path_w_length[idx_ref, -1]
            if q_ref <= q_curr:
                idx0 = idx_ref
                idx1 = min(idx_ref + 1, n_size-1)
            else:
                idx0 = max(idx_ref - 1, 0)
                idx1 = idx_ref
            # idx0_, idx1_ = np.argsort(np.abs(q_curr - path_w_length[:, -1]))[:2]
            # if idx0_ < idx1_:
            #     idx0 = idx0_
            #     idx1 = idx1_
            # else:
            #     idx0 = idx1_
            #     idx1 = idx0_
            s_inter = (q_curr - path_w_length[idx0, -1]) / (path_w_length[idx1, -1] - path_w_length[idx0, -1])
            s_trj.append(s_inter)

            trj_gripper[i, :3] = path_w_length[idx0, :3] + s_inter * (path_w_length[idx1, :3] - path_w_length[idx0, :3])
            cos_0 = np.cos(path_w_length[idx0, 3])
            cos_1 = np.cos(path_w_length[idx1, 3])
            sin_0 = np.sin(path_w_length[idx0, 3])
            sin_1 = np.sin(path_w_length[idx1, 3])
            trj_gripper[i, 3] = np.arctan2(sin_0 + s_inter * (sin_1 - sin_0), cos_0 + s_inter * (cos_1 - cos_0))

    global TRJ_GRIPPER
    TRJ_GRIPPER = trj_gripper

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(trj_gripper[:, 0], trj_gripper[:, 1], trj_gripper[:, 2], 'b--')
    ax.plot3D(path_w_length[:, 0], path_w_length[:, 1], path_w_length[:, 2], 'g--')
    for i in range(0, trj_gripper.shape[0], 40):
        ax.quiver(trj_gripper[i, 0], trj_gripper[i, 1], trj_gripper[i, 2],
                  np.cos(trj_gripper[i, 3]), np.sin(trj_gripper[i, 3]), 0, length=1, normalize=True)
    plt.draw()
    plt.pause(0.1)

    return 1


def compute_rope_orientation_at_idx(rope_positions, idx):
    if idx == 0:
        return np.arctan2(rope_positions[idx, 1]-rope_positions[idx+1, 1], rope_positions[idx, 0]-rope_positions[idx+1, 0])
    else:
        return np.arctan2(rope_positions[idx-1, 1]-rope_positions[idx, 1], rope_positions[idx-1, 0]-rope_positions[idx, 0])


def follow_trajectory(gripper):
    global TRJ_GRIPPER, TRJ_IDX
    if TRJ_IDX is None:
        TRJ_IDX = 0
    else:
        TRJ_IDX += 1

    '''
    plt.figure()
    yaw_gripper_space = np.zeros(TRJ_GRIPPER.shape[0])
    for j in range(yaw_gripper_space.size):
        yaw_gripper_space[j] = task_2_gripper_angle(task_angle=TRJ_GRIPPER[j, 3])
    plt.plot(TRJ_GRIPPER[:, 3], 'g--')
    plt.plot(yaw_gripper_space, 'r--')
    plt.draw()
    plt.pause(0.1)
    '''

    if TRJ_IDX < TRJ_GRIPPER.shape[0]:
        new_pose = X_GRIPPER.copy()
        new_pose[:3] = TRJ_GRIPPER[TRJ_IDX, :3]
        # yaw_task_space = np.arctan2(np.sin(TRJ_GRIPPER[TRJ_IDX, 3]), np.cos(TRJ_GRIPPER[TRJ_IDX, 3]))
        yaw_gripper_space = task_2_gripper_angle(TRJ_GRIPPER[TRJ_IDX, 3])  # angle_transform(angle_in=yaw_task_space, forward=1)
        new_pose[4] = yaw_gripper_space
        gripper.set_pose(new_pose)
    else:
        TRJ_IDX = None
        return 1
    return 0


def plan_and_execute(gripper, path):
    global CURRENT_STEP, PATH_CONCAT, X_GRIPPER
    v_gripper = 2*16
    a_gripper = 2*8

    # First add current pose of left gripper to path
    if CURRENT_STEP is None:
        x_gripper = X_GRIPPER.copy()

        first_point = np.asarray([x_gripper[0], x_gripper[1], path[0, 2], path[0, 3]]).reshape(1, -1)
        path_concat = np.concatenate([first_point, path], axis=0)
        PATH_CONCAT = path_concat
        CURRENT_STEP = 0

    if CURRENT_STEP == 0:
        if compute_continuous_trajectory(v_gripper=v_gripper, a_gripper=a_gripper, path=PATH_CONCAT):
            CURRENT_STEP += 1
    if CURRENT_STEP == 1:
        if follow_trajectory(gripper=gripper):
            CURRENT_STEP = None
            return 1
    return 0


class Peg(RigidObject):
    def set_color(self, new_color: Tuple[int, int, int]):
        set_color(self.visual_model_node.OglModel, color=tuple(color / 255 for color in new_color))


# RopeAccessController reads object state via interrupt
class RopeAccessController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rope = kwargs.get("rope")

    # onAnimateEndEvent-method is interrupt method that is triggered after each time step
    def onAnimateEndEvent(self, event):
        global ROPE_POSITIONS, PROGRAM_STEP, ROPE_POSITIONS_HISTORY
        ROPE_POSITIONS = self.rope.get_positions()


# GripperAccessController controls moveable left gripper
class GripperAccessController(Sofa.Core.Controller):

    # Insert desired waypoints as class attributes
    def __init__(self, waypoints, add_args, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.waypoints = waypoints
        self.gripper = kwargs.get("gripper")

    # onAnimateEndEvent-method is interrupt method that is triggered after each time step
    # global variables are used to monitor state of program
    def onAnimateEndEvent(self, event):
        dt = 0.01
        global STATE, X_GRIPPER, X_GRIPPER_LAST, V_GRIPPER

        X_GRIPPER_LAST = X_GRIPPER
        X_GRIPPER = self.gripper.get_pose()
        #print(get_gripper_pose(self.gripper))
        roll = quaternion_to_euler_angles(X_GRIPPER[3:])

        print(roll)
        print(self.waypoints)

        if X_GRIPPER_LAST is not None and X_GRIPPER is not None:
            V_GRIPPER = (X_GRIPPER - X_GRIPPER_LAST) / dt

        if STATE == 0:
            if pause(steps=100):
                STATE += 1

        if STATE == 1:
            if plan_and_execute(gripper=self.gripper, path=self.waypoints):
                STATE += 1

        if STATE == 2:
            print("Program finished")
            STATE += 1


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    # eye_config: List[Tuple[int, int, int, int]] = DEFAULT_EYE_CONFIG,
    # eye_reset_noise: Union[None, Dict[str, np.ndarray], List[Dict[str, np.ndarray]]] = None,
    randomize_gripper: bool = False,
    randomize_grasp_index: bool = False,
    start_grasped: bool = True,
    debug_rendering: bool = False,
    positioning_camera: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    camera_pose: np.ndarray = np.array([115.0, -60.0, 75.0, 0.42133736, 0.14723759, 0.29854957, 0.8436018]),
    # another sensible camera pose with broader view
    # [50.0, -75.0, 150.0, 0.3826834, 0.0, 0.0, 0.9238795]
    no_textures: bool = False,
    # waypoints_xyz: np.ndarray = np.asarray([[25, 45], [45, 25], [75, 30], [85, 30], [95, 20], [85, 10], [65, 10]]),
    # waypoints_yaw: np.ndarray = np.asarray([5/4 * np.pi, 7/4 * np.pi, 15/8 * np.pi, 7/4 * np.pi, 6/4 * np.pi, 5/4 * np.pi, 4/4 * np.pi]),
    waypoints: np.ndarray = np.zeros(4),
    contact_pos: np.ndarray = np.zeros(2),
    add_args: list = []
) -> Dict:
    gravity = -981  # mm/s^2

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        constraint_solver=ConstraintSolverType.GENERIC,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        scene_has_collisions=True,
        collision_detection_method=IntersectionMethod.MINPROXIMITY,
        collision_detection_method_kwargs={
            "alarmDistance": 1.0,
            "contactDistance": 0.05,
        },
        scene_has_cutting=False,
        contact_friction=0.0,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    # Initialize rope
    # If change in rope length desired: Change rope length and start_position[0] in l. 499,
    # Change initial pose of left gripper rcm_pose[0] in l. 559, change index to grasp for right gripper grasp_index_pair[1] in l. 634
    poses_for_rope = poses_for_linear_rope(length=210, num_points=100, start_position=np.array([0.0, 50.0, 19.0]))

    # Rotate the rope such that y axis of gripper and rope align
    transformation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([180.0, 0.0, 0.0])))
    poses_for_rope = [np.append(pose[:3], multiply_quaternions(transformation_quaternion, pose[3:])) for pose in poses_for_rope]

    # Check here to get attributes of rope
    rope = TransferRope(
        parent_node=scene_node,
        name="rope",
        waypoints=np.asarray([0.0, 0., 0.]),
        radius=0.8,  # 1.0,
        beam_radius=2.5,  # 2.0,
        total_mass=1 * 5.0,
        poisson_ratio=0.5,
        young_modulus=0.3 * 5e4,
        fix_start=False,
        fix_end=False,
        collision_type=RopeCollisionType.SPHERES,
        animation_loop_type=animation_loop,
        mechanical_damping=1.0,
        show_object=debug_rendering,
        show_object_scale=3,
        poses=poses_for_rope,
    )

    cartesian_workspace = {
        "low": np.array([-20.0, -20.0, 0.0]),
        "high": np.array([230.0, 180.0, 100.0]),
    }

    board_node = scene_node.addChild("board")

    board_limits = {
        "low": np.array([5., 5., -5.]),
        "high": np.array([55.+40., 65.+40., 0.]),
    }
    grid_shape = (10, 10, 2)
    board_node.addObject("RegularGridTopology", min=board_limits["low"], max=board_limits["high"], n=grid_shape)
    if no_textures:
        board_node.addObject("OglModel", color=[value / 255 for value in [246.0, 205.0, 139.0]])
    else:
        board_texture_path = TEXTURE_DIR / "wood_texture.png"
        board_node.addObject("OglModel", texturename=str(board_texture_path))
        board_node.init()
        # Fix the texture coordinates
        with board_node.OglModel.texcoords.writeable() as texcoords:
            for index, coordinates in enumerate(texcoords):
                x, y, _ = np.unravel_index(index, grid_shape, "F")
                coordinates[:] = [x / grid_shape[0], y / grid_shape[1]]

    # Left gripper has rope grasped at moveable end and can be manipulated via GripperAccessController
    left_gripper = ArticulatedGripper(
        parent_node=scene_node,
        name="left_gripper",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
        visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
        rope_to_grasp=rope,
        ptsd_state=np.array([0.0, 0.0, 180.0, 50.0]),
        # Initialize gripper state [gripper x, gripper y, gripper z, unused angle, yaw, unused angle]
        rcm_pose=np.array([4.0, 50.0, 85.0, 0.0, 180.0, 0.0]),
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "backside": [[0, -1.5, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        jaw_length=25,
        angle=5.0,
        angle_limits=(0.0, 60.0),
        total_mass=1e12,
        mechanical_binding=MechanicalBinding.SPRING,
        animation_loop_type=animation_loop,
        show_object=debug_rendering,
        show_object_scale=10,
        show_remote_center_of_motion=debug_rendering,
        state_limits={
            "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0]),
            "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 100.0]),
        },
        spring_stiffness=1e29,
        angular_spring_stiffness=1e29,
        articulation_spring_stiffness=1e29,
        spring_stiffness_grasping=1e9,
        angular_spring_stiffness_grasping=1e9,
        angle_to_grasp_threshold=10.0,
        angle_to_release_threshold=15.0,
        collision_group=0,
        collision_contact_stiffness=100,
        cartesian_workspace=cartesian_workspace,
        start_grasped=start_grasped,
        grasp_index_pair=(5, 2),  # (5, 2),
        ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]) if randomize_gripper else None,
        angle_reset_noise=20.0 if randomize_gripper else None,
        deactivate_collision_while_grasped=True,
    )

    # Right gripper as rope gripped at fixed end and is not moveable
    right_gripper = ArticulatedGripper(
        parent_node=scene_node,
        name="right_gripper",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
        visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
        rope_to_grasp=rope,
        ptsd_state=np.array([0.0, 0.0, 180.0, 50.0]), # np.array([0.0, 0.0, 180.0, 50.0]),
        # Initialize gripper state [gripper x, gripper y, gripper z, unused angle, yaw, unused angle]
        rcm_pose=np.array([100., 50., 85., 0.0, 180., 0.0]),
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)], # [[0, 0, 5 + i * 2] for i in range(10)],
            "backside": [[0, -1.5, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        jaw_length=25,
        angle=5.0,
        angle_limits=(0.0, 60.0),
        total_mass=1e12,
        mechanical_binding=MechanicalBinding.SPRING,
        animation_loop_type=animation_loop,
        show_object=debug_rendering,
        show_object_scale=10,
        show_remote_center_of_motion=debug_rendering,
        state_limits={
            "low": np.array([-90, -90, np.iinfo(np.int16).min, 0]),
            "high": np.array([90, 90, np.iinfo(np.int16).max, 100]),
        },
        spring_stiffness=1e29,
        angular_spring_stiffness=1e29,
        articulation_spring_stiffness=1e29,
        spring_stiffness_grasping=1e9,
        angular_spring_stiffness_grasping=1e9,
        angle_to_grasp_threshold=10.0,
        angle_to_release_threshold=15.0,
        collision_group=0,
        collision_contact_stiffness=100,
        cartesian_workspace=cartesian_workspace,
        start_grasped=start_grasped,
        grasp_index_pair=(5, 48), # 65+2),
        grasp_index_reset_noise={"low": -5, "high": 15} if randomize_grasp_index else None,
        ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]) if randomize_gripper else None,
        angle_reset_noise=None,
        deactivate_collision_while_grasped=True,
    )

    if not positioning_camera:
        scene_node.addObject(left_gripper)
        scene_node.addObject(right_gripper)

    contact_listeners = {"left_gripper": [], "right_gripper": []}
    contact_listener_info = {"left_gripper": [], "right_gripper": []}

    add_peg_collision = partial(
        add_collision_model,
        contact_stiffness=1e2,
        model_types=[CollisionModelType.LINE, CollisionModelType.TRIANGLE],
        is_static=True,
        collision_group=1,
    )
    peg_visual_surface_mesh_path = HERE / "meshes/peg.stl"
    peg_collision_surface_mesh_path = HERE / "meshes/peg_collision.stl"

    # Initialize contacts
    '''''''''
    pegs = []
    peg_x = contact_pos[:, 0]
    peg_y = contact_pos[:, 1]
    peg_z = 5 * np.ones(contact_pos.shape[1])
    for i in range(peg_x.size):
        pegs.append(
            Peg(
                parent_node=scene_node,
                name="peg0",
                pose=(peg_x[i], peg_y[i], peg_z[i]) + (1.0, 0.0, 0.0, 1.0),  # (100, 60, 5) + (1.0, 0.0, 0.0, 1.0),
                visual_mesh_path=peg_visual_surface_mesh_path,
                collision_mesh_path=peg_collision_surface_mesh_path,
                animation_loop_type=animation_loop,
                add_collision_model_func=add_peg_collision,
                fixed_position=True,
                fixed_orientation=True,
                total_mass=1.0,
            )
        )
    '''''''''

    rope.node.addObject(
        "PlaneForceField",
        normal=[0, 0, 1],
        d=rope.radius,
        stiffness=300,
        damping=1,
        showPlane=True,
        showPlaneSize=0.1,
    )

    scene_creation_result = {
        "interactive_objects": {
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
        },
        "rope": rope,
    }

    # Initialze controller that work via interrupts
    root_node.addObject(RopeAccessController('RopeAccessor', name='ropeAccessor', rope=rope))
    root_node.addObject(GripperAccessController(waypoints, 'GripperAccessorLeft', name='gripperAccessorLeft', gripper=left_gripper))

    return rope.get_positions()

