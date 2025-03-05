import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot


def convert_qdof_adr(model, joint_ids=None, concat=False):
    """Convert joint ids to qpos_dofs. A list of qpos_dofs is associated with each
    joint id; hence, this function returns a list of lists of qposard."""
    # WARNING: I'm not totally sure that len(bodyid) is always the number of
    # qdofs for a joint, which I assume here.
    if joint_ids is None:
        joint_ids = range(model.njnt)
    if not hasattr(joint_ids, "__len__"):
        joint = model.joint(joint_ids)
        qpos_st = joint.dofadr.item()
        n_qposs = len(joint.bodyid)
        qpos_dofs = list(range(qpos_st, qpos_st + n_qposs))
        return qpos_dofs
    qpos_dofs = []
    for id in joint_ids:
        joint = model.joint(id)
        qpos_st = joint.dofadr.item()
        n_qposs = len(joint.bodyid)
        qpos_dof = list(range(qpos_st, qpos_st + n_qposs))
        if concat:
            qpos_dofs.extend(qpos_dof)
        else:
            qpos_dofs.append(qpos_dof)
    return qpos_dofs


def convert_qpos_adr(model, joint_ids=None, concat=False):
    """Convert joint ids to qpos_adr. A list of qpos_adr is associated with each
    joint id; hence, this function returns a list of lists of qposadr."""
    if joint_ids is None:
        joint_ids = range(model.njnt)
    singleton = False
    if not hasattr(joint_ids, "__len__"):
        singleton = True
        joint_ids = [joint_ids]
    qposadrs = []
    for id in joint_ids:
        joint = model.joint(id)
        if len(joint.jntid) == 1:  # Simple 1-d joint
            qposadr = joint.qposadr.tolist()
        elif len(joint.jntid) == 6:  # quaternion joint
            qposadr = list(range(joint.qposadr.item(), joint.qposadr.item() + 7))
        else:
            raise ValueError("Joint with unsupported DOF number.")
        if concat or singleton:
            qposadrs.extend(qposadr)
        else:
            qposadrs.append(qposadr)
    return qposadrs


def get_body_qposaddr(model, body_ids=None):
    # adapted to mujoco 2.3+
    if body_ids is None:
        body_ids = range(model.nbody)
    body_qposaddr = dict()
    for i in body_ids:
        body_name = model.body(i).name
        # body_jntadr: start addr of joints; -1: no joints
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        # body_jntnum: number of joints for this body
        end_joint = start_joint + model.body_jntnum[i]
        # jnt_qposadr: start addr in 'qpos' for joint's data
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            # nq: number of generalized coordinates = dim(qpos)
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def get_body_qpos_list(model, body_ids=None):
    body_qposaddr = get_body_qposaddr(model, body_ids)
    qpos_list = []
    for qpos_idx_pair in body_qposaddr.values():
        qpos_list.extend(list(range(qpos_idx_pair[0], qpos_idx_pair[1])))
    return qpos_list


def get_body_qveladdr(model, body_ids=None):
    # adapted to mujoco 2.3+
    if body_ids is None:
        body_ids = range(model.nbody)
    body_qveladdr = dict()
    for i in body_ids:
        body_name = model.body(i).name
        # body_jntadr: start addr of joints; -1: no joints
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        # body_jntnum: number of joints for this body
        end_joint = start_joint + model.body_jntnum[i]
        start_qveladdr = model.jnt_dofadr[start_joint]
        if end_joint < len(model.jnt_dofadr):
            end_qveladdr = model.jnt_dofadr[end_joint]
        else:
            end_qveladdr = model.nv
        body_qveladdr[body_name] = (start_qveladdr, end_qveladdr)
    return body_qveladdr


def get_body_qvel_list(model, body_ids=None):
    body_qveladdr = get_body_qveladdr(model, body_ids)
    qvel_list = []
    for qvel_idx_pair in body_qveladdr.values():
        qvel_list.extend(list(range(qvel_idx_pair[0], qvel_idx_pair[1])))
    return qvel_list


def get_jnt_range(model):
    jnt_range = dict()
    for i in range(model.njnt):
        if i == model.njnt - 1:
            end_p = model.name_geomadr[0]
        else:
            end_p = model.name_jntadr[i + 1]
        name = model.names[model.name_jntadr[i] : end_p].decode("utf-8").rstrip("\x00")
        jnt_range[name] = model.jnt_range[i]
    return jnt_range


# def get_actuator_names(model):
# actuators = []
# for i in range(model.nu):
# if i == model.nu - 1:
# end_p = None
# for el in ["name_sensoradr", "name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr"]:
# v = getattr(model, el)
# if np.any(v):
# end_p = v[0]
# if end_p is None:
# end_p = model.nnames
# else:
# end_p = model.name_actuatoradr[i+1]
# name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
# actuators.append(name)
# return actuators


def get_actuator_names(model):
    actuators = [model.actuator(k).name for k in range(model.nu)]
    return actuators


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def add_visual_rbox(scene, point1, point2, rgba):
    """Adds one rectangle to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_BOX,
        0.01,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )
