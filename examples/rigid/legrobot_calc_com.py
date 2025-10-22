import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene()
robot = scene.add_entity(
    gs.morphs.URDF(file="./genesis/assets/urdf/legrobot/urdf/legrobot.urdf", pos=(0.0, 0.0, 0.6), fixed=True),   # fixed=Trueでsim空間に固定
)

def compute_robot_com(robot):
    total_mass = 0.0
    weighted_com = np.zeros(3)

    for link in robot.links:
        m = link.mass
        if m <= 0:
            continue

        # リンクローカル座標での CoM
        local_com = np.array(link.com)

        # ワールド座標に変換
        world_com = link.pose.transform_point(local_com)

        weighted_com += m * world_com
        total_mass += m

    return weighted_com / total_mass if total_mass > 0 else np.zeros(3)

scene.build()

com = compute_robot_com(robot)
print(f"Center of Mass: {com}")

# シミュレーションループ
for _ in range(100):
    scene.step()
    print("CoM:", compute_robot_com(robot))


'''
ChatGPTに効いて書いたが動かない。。
'''