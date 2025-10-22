import argparse

import numpy as np

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, 1),
        ),
    )

    ########################## entities ##########################

    robot = scene.add_entity(
        gs.morphs.URDF(file="./genesis/assets/urdf/clala/urdf/clala_stl.urdf", fixed=True),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    center = np.array([0.05, -0.05, -0.25])
    center2 = np.array([0.05, 0.05, -0.25])
    r = 0.05

    ee_link = robot.get_link("R_toe")
    print(ee_link.name, ee_link.idx)
    ee_link2 = robot.get_link("L_toe")
    print(ee_link2.name, ee_link2.idx)
    ee_links = [ee_link, ee_link2]
    for i in range(0, 2000):
        # target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        target_entity.set_qpos(np.concatenate([center, target_quat]))
        q, err = robot.inverse_kinematics_multilink(
            links=[ee_link, ee_link2],
            poss=[center, center2],
            quats=[target_quat, target_quat],
            return_error=True,
            rot_mask=[False, False, False],  # for demo purpose: only care about direction of z-axis
        )
        print("error:", err)

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        robot.set_qpos(q)
        scene.visualizer.update()


if __name__ == "__main__":
    main()
