import numpy as np

_FLOAT_EPS = np.finfo(np.float64).eps


def distance_between_objects(pos1, pos2):
    return abs(np.linalg.norm(pos1 - pos2))


def ball_in_hole(ball_pos, hole_pos):
    return (distance_between_objects(ball_pos, hole_pos) < 0.06).astype(np.float32)


def check_grasp(env):
    club_body_id = env.golf_club_id
    left_finger_body_id = env.left_finger_body_id
    right_finger_body_id = env.right_finger_body_id

    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False

    contact = env.robot_model.data.contact

    club_left_contact = False
    club_right_contact = False

    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2

        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]

        if body1 == club_body_id or body2 == club_body_id:
            other_body = body2 if body1 == club_body_id else body1

            if other_body == left_finger_body_id:
                club_left_contact = True
            elif other_body == right_finger_body_id:
                club_right_contact = True

    return club_left_contact and club_right_contact


def check_ball_club_contact(env):
    club_body_id = env.club_head_id
    ball_body_id = env.golf_ball_id
    ncon = env.robot_model.data.ncon
    if ncon == 0:
        return False

    contact = env.robot_model.data.contact

    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2

        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]

        if (body1 == club_body_id and body2 == ball_body_id) or (
            body1 == ball_body_id and body2 == club_body_id
        ):
            return True

    return False


def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0

    if not eval_state.get("closest_distance_to_club", False):
        eval_state["closest_distance_to_club"] = 10000

    ee_pos = env.robot_model.data.site(env.ee_site_id).xpos

    club_grip_pos = env.robot_model.data.xpos[env.golf_club_id]

    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]

    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    ee_club_dist = distance_between_objects(ee_pos, club_grip_pos)

    robot_grasped_club = check_grasp(env)

    robot_hit_ball_with_club = check_ball_club_contact(env)

    if robot_hit_ball_with_club and not eval_state.get(
        "robot_hit_ball_with_club", False
    ):
        eval_state["robot_hit_ball_with_club"] = eval_state["timestep"]

    if robot_grasped_club and not eval_state.get("robot_grasped_club", False):
        eval_state["robot_grasped_club"] = eval_state["timestep"]

    eval_state["closest_distance_to_club"] = min(
        eval_state["closest_distance_to_club"], ee_club_dist
    )

    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0

        if ball_in_hole(ball_pos, hole_pos):
            reward += (
                10.0 - eval_state["timestep"] * 0.01
            )  # Min 4.5 reward for getting the ball in the hole
        else:
            reward += (
                1.1867 - distance_between_objects(ball_pos, hole_pos)
            ) * 2  # Max ~2.25 reward for being close to the hole

        if eval_state.get("robot_hit_ball_with_club", False):
            reward += (
                3.5 - eval_state["robot_hit_ball_with_club"] * 0.005
            )  # Min 0.25 reward for hitting the ball with club head

        if eval_state.get("robot_grasped_club", False):
            reward += (
                1.65 - eval_state["robot_grasped_club"] * 0.001
            )  # Min 1.0 reward for grasping
        else:
            reward += max(
                1 - eval_state["closest_distance_to_club"], 0
            )  # Max 1.0 reward for being close to the club grip

        return (reward, eval_state)

    return (0.0, eval_state)
