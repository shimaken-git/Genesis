import argparse

import genesis as gs

import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 1.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.0167,
            constraint_solver=gs.constraint_solver.Newton,  #defaultではCG(conjugate gradian共役勾配法)
            enable_joint_limit=True,
            enable_collision=True,    #defaultでTrue
            enable_self_collision=True,   #defaultでFalse
            gravity=(0, 0, -9.8),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(file="./genesis/assets/urdf/clala/urdf/clala_stl_2.urdf", pos=(0.0, 0.0, 0.5), fixed=True),   # fixed=Trueでsim空間に固定
    )
    robot2 = scene.add_entity(
        gs.morphs.URDF(
            file="./genesis/assets/urdf/clala/urdf/clala_stl.urdf",
            pos=(0.0, 0.75, 0.5), fixed=True
        ),
        material=gs.materials.Rigid(gravity_compensation=0.0),
    )
    robot3 = scene.add_entity(
        gs.morphs.URDF(file="./genesis/assets/urdf/clala/urdf/clala_stl.urdf", pos=(0.0, 1.5, 0.5), fixed=True),
        material=gs.materials.Rigid(gravity_compensation=0.0),    #gravity_compensation=1.0で無重力
    )

    scene.build()

    print(robot)    # print(scene.entities[1])と同じ

    dof_names = ["L_calf_joint", "L_hip2_joint", "L_hip_joint", "L_thigh_joint", "L_toe_joint", "R_calf_joint", "R_hip2_joint", "R_hip_joint", "R_thigh_joint", "R_toe_joint",]

    motor_dofs = [robot.get_joint(name).dof_idx_local for name in dof_names]
    print(motor_dofs)
    ini_positions = [0.0] * 10
    # ini_positions = [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0]   # calf
    # positions = [1.5, 0.0, 0.0, -0.75, -0.75, 1.5, 0.0, 0.0, -0.75, -0.75]
    positions = [0.0, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, -1.2, 0.0]   # thigh
    # positions = [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, -0.7, 0.0, 0.0, 0.0]   # hip2
    # positions = [1.5, 0.0, 0.0, -0.5, 0.0, 1.5, 0.0, 0.0, -0.5, 0.0]   # calf
    robot.set_dofs_kp([30.0] * 10, motor_dofs)
    robot.set_dofs_kv([0.5] * 10, motor_dofs)
    robot.set_dofs_damping([0.1] * 10, motor_dofs)   #動きにくさ
    robot.set_dofs_armature([0.1] * 10, motor_dofs)  #armatture inertia 電機子イナーシャ  止まりにくさ
    robot.set_dofs_stiffness([0.0] * 10, motor_dofs)
    robot.control_dofs_position(ini_positions, motor_dofs)

    robot2.set_dofs_kp([30.0] * 10, motor_dofs)
    robot2.set_dofs_kv([0.5] * 10, motor_dofs)
    robot2.set_dofs_damping([0.1] * 10, motor_dofs)
    robot2.set_dofs_armature([0.0] * 10, motor_dofs)
    robot2.set_dofs_stiffness([0.0] * 10, motor_dofs)
    robot2.control_dofs_position(ini_positions, motor_dofs)

    # thigh
    robot3.set_dofs_kp([29.0] * 10, motor_dofs)
    robot3.set_dofs_kv([0.795] * 10, motor_dofs)
    robot3.set_dofs_damping([1.0] * 10, motor_dofs)
    robot3.set_dofs_armature([0.00567] * 10, motor_dofs)
    robot3.set_dofs_stiffness([0.0] * 10, motor_dofs)
    robot3.control_dofs_position(ini_positions, motor_dofs)
    # hip2
    # robot3.set_dofs_kp([9.1] * 10, motor_dofs)
    # robot3.set_dofs_kv([0.758] * 10, motor_dofs)
    # robot3.set_dofs_damping([0.1] * 10, motor_dofs)
    # robot3.set_dofs_armature([0.021] * 10, motor_dofs)
    # robot3.set_dofs_stiffness([0.0] * 10, motor_dofs)
    # robot3.control_dofs_position(ini_positions, motor_dofs)
    # calf
    # robot3.set_dofs_kp([15.3] * 10, motor_dofs)
    # robot3.set_dofs_kv([0.934] * 10, motor_dofs)
    # robot3.set_dofs_damping([0.1] * 10, motor_dofs)
    # robot3.set_dofs_armature([0.0173] * 10, motor_dofs)
    # robot3.set_dofs_stiffness([0.0] * 10, motor_dofs)
    # robot3.control_dofs_position(ini_positions, motor_dofs)

    counter = 0
    count2 = 0
    p = []
    v = []
    t = []
    for i in range(400):
        if counter == 100:
            if count2 % 2 == 0 :
                robot.control_dofs_position(positions, motor_dofs)
                robot2.control_dofs_position(positions, motor_dofs)
                robot3.control_dofs_position(positions, motor_dofs)
            else:
                robot.control_dofs_position(ini_positions, motor_dofs)
                robot2.control_dofs_position(ini_positions, motor_dofs)
                robot3.control_dofs_position(ini_positions, motor_dofs)
            counter = 0
            # count2 += 1
        counter += 1
        scene.step()
        t.append(i * 0.0167)
        p.append(robot2.get_dofs_position(motor_dofs)[3].item())   # 3: thigh 1: hip2 0: calf
        v.append(robot2.get_dofs_velocity(motor_dofs)[3].item())
        print(robot2.get_pos())
        print(robot2.get_links_pos())
        link = robot2.get_link(name='trunk')
        print(link.name)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(t, p)
    ax2.plot(t, v)
    plt.show()
    with open("step_response_sim.txt","w") as o:
        for d in p:
            print(d, file=o) 
        for d in v:
            print(d, file=o) 
    o.close()

if __name__ == "__main__":
    main()
