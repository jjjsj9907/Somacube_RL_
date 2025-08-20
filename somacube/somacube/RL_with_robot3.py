import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque, namedtuple
import time
from scipy.spatial.transform import Rotation as R 
from scipy.spatial.transform import Rotation
from od_msg.srv import SrvDepthPosition
import os
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import DR_init
from somacube.onrobot import RG
import sys

PACKAGE_NAME = "somacube"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

CUBE_MODEL_FILENAME = "best_soma_model_physical.pth"

CUBE_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", CUBE_MODEL_FILENAME)


# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
UP_JOG = [8.81, 5.70, 59.50, -7.02, 90.07, -6.1]
# ANSWER_POINT = [424.850, 78.830, 12.4] # 총 7.5cm 개당 2.5 cm
# ANSWER_POINT = [449.850, 53.830, 100]
ANSWER_POINT = [437.35, 16.33, 100]

FORCE_VALUE = 10


tool_dict = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7"}

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait,\
                           trans, wait, DR_BASE, amovel, amovej, \
                        get_current_solution_space, get_current_posj,\
                        movejx, \
                        release_compliance_ctrl,\
                        release_force,\
                        check_force_condition,\
                        task_compliance_ctrl,\
                        set_desired_force,\
                        DR_FC_MOD_REL, DR_AXIS_Z,ikin, fkin
    from DR_common2 import posx, posj
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup. Do not modify this area ############

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)


########### Robot Controller ############


re_grap_pos = [624.960, 119.680, -22.780, 68.28, -179.3, 67.66]
HOME = [0, 0, 90, 0, 90, 0]
def up_pos(set_pos, axis, val):
        pos = set_pos.copy()
        pos[axis] += val
        return posx(pos)


def is_problematic_solution(joints, threshold_checks=True):
    """
    실제 로봇의 조인트 한계를 적용한 솔루션 검증
    
    조인트 한계:
    - Joint 1: -360 ~ 360도
    - Joint 2: -95 ~ 95도  
    - Joint 3: -135 ~ 135도
    - Joint 4: -360 ~ 360도
    - Joint 5: -135 ~ 135도
    - Joint 6: -360 ~ 360도
    """
    if joints is None or len(joints) != 6:
        return True
    
    # 실제 로봇 조인트 한계 정의
    joint_limits = [
        (-360, 360),   # Joint 1
        (-95, 95),     # Joint 2
        (-135, 135),   # Joint 3
        (-360, 360),   # Joint 4
        (-135, 135),   # Joint 5
        (-360, 360)    # Joint 6
    ]
    
    # 각 조인트가 한계 내에 있는지 확인
    for i, (joint_val, (min_limit, max_limit)) in enumerate(zip(joints, joint_limits)):
        if joint_val < min_limit or joint_val > max_limit:
            print(f"조인트 {i+1}이 한계를 벗어남: {joint_val:.1f}° (한계: {min_limit}° ~ {max_limit}°)")
            return True
    
    # 추가 안전성 검사: 극소값 체크 (선택적)
    if threshold_checks:
        for i, angle in enumerate(joints):
            if abs(angle) < 0.1:  # 0.1도 이하 극소값
                print(f"조인트 {i+1}에서 극소값 감지: {angle:.6f}°")
                return True
    
    return False


def select_safe_joint_solution(target_pos_rotation, preference="elbow_down", avoid_problematic=True):
    """
    안전성을 우선하는 간단한 솔루션 선택기 (문제 있는 솔루션 필터링 추가)
    
    Args:
        target_pos_rotation: 목표 위치/자세
        preference: "elbow_down", "elbow_up", "closest" 중 선택
        avoid_problematic: 문제 있는 솔루션 회피 여부
    
    Returns:
        best_joints: 선택된 조인트 각도
    """
    current_joints = get_current_posj()
    solutions = []
    
    # 모든 솔루션 수집 및 필터링
    for i in range(8):
        try:
            joints = ikin(target_pos_rotation, i, DR_BASE)
            if joints is not None and len(joints) == 6:
                # 문제 있는 솔루션 필터링
                if avoid_problematic and is_problematic_solution(joints):
                    print(f"솔루션 {i} 제외됨 (문제 있는 각도)")
                    continue
                solutions.append((i, joints))
                print(f"솔루션 {i} 유효: {joints}")
        except:
            print(f"솔루션 {i}: 계산 실패")
            continue
    
    if not solutions:
        print("안전한 솔루션이 없습니다. 필터링 없이 재시도합니다.")
        return select_safe_joint_solution(target_pos_rotation, preference, avoid_problematic=False)
    
    if preference == "closest":
        # 현재 위치에서 가장 가까운 솔루션
        best_distance = float('inf')
        best_solution = None
        best_idx = -1
        
        for sol_idx, joints in solutions:
            distance = np.sum(np.abs(np.array(joints) - np.array(current_joints)))
            if distance < 20:
                continue # 일정 거리 밑이면 실패한 각도로 취급
            if distance < best_distance:
                best_distance = distance
                best_solution = joints
                best_idx = sol_idx
        
        print(f"Closest 솔루션 선택: {best_idx}, 거리: {best_distance:.1f}")
        return best_solution
    
    elif preference == "elbow_down":
        # 팔꿈치 아래쪽 솔루션 선호
        elbow_down_solutions = [s for s in solutions if s[1][2] > 0]
        if elbow_down_solutions:
            # 그 중 가장 가까운 것
            best_distance = float('inf')
            best_solution = None
            best_idx = -1
            for sol_idx, joints in elbow_down_solutions:
                distance = np.sum(np.abs(np.array(joints) - np.array(current_joints)))
                if distance < best_distance:
                    best_distance = distance
                    best_solution = joints
                    best_idx = sol_idx
            print(f"Elbow down 솔루션 선택: {best_idx}, 거리: {best_distance:.1f}")
            return best_solution
        else:
            # 팔꿈치 아래 솔루션이 없으면 가장 가까운 것
            print("팔꿈치 아래 솔루션 없음, closest로 변경")
            return select_safe_joint_solution(target_pos_rotation, "closest", avoid_problematic)
    
    elif preference == "elbow_up":
        # 팔꿈치 위쪽 솔루션 선호
        elbow_up_solutions = [s for s in solutions if s[1][2] <= 0]
        if elbow_up_solutions:
            best_distance = float('inf')
            best_solution = None
            best_idx = -1
            for sol_idx, joints in elbow_up_solutions:
                distance = np.sum(np.abs(np.array(joints) - np.array(current_joints)))
                if distance < best_distance:
                    best_distance = distance
                    best_solution = joints
                    best_idx = sol_idx
            print(f"Elbow up 솔루션 선택: {best_idx}, 거리: {best_distance:.1f}")
            return best_solution
        else:
            print("팔꿈치 위 솔루션 없음, closest로 변경")
            return select_safe_joint_solution(target_pos_rotation, "closest", avoid_problematic)
    
    # 기본값: 첫 번째 유효한 솔루션
    print(f"기본 솔루션 선택: {solutions[0][0]}")
    return solutions[0][1]


def choose_best_orientation(start_pose, zyz_base_delta):
    """
    AI가 제안한 회전(zyz_base_delta)을 기반으로 두 개의 후보 자세를 만듭니다.
    1. 원래 자세
    2. 툴 롤을 90도 추가한 자세
    그 중 '손 날'이 더 수평에 가까운(안전한) 자세를 선택하여 반환합니다.
    """
    # 후보 1: AI가 제안한 원래 목표 자세 계산
    pose1 = apply_rotation_manually(start_pose, zyz_base_delta)

    # 후보 2: 툴 롤(Rz')을 90도 추가한 목표 자세 계산
    correction_delta = zyz_base_delta.copy()
    correction_delta[2] += 90.0
    pose2 = apply_rotation_manually(start_pose, correction_delta)
    
    # "손 날"의 방향을 나타내는 그리퍼의 X축([1,0,0])이 베이스 기준에서 어떤 방향인지 계산
    r1 = R.from_euler('zyz', pose1[3:], degrees=True)
    gripper_x_vector1 = r1.apply([1, 0, 0])
    
    r2 = R.from_euler('zyz', pose2[3:], degrees=True)
    gripper_x_vector2 = r2.apply([1, 0, 0])
    
    # "손 날"의 Z값(수직 성분)의 절대값이 작을수록 더 수평에 가깝고 안전함
    score1 = abs(gripper_x_vector1[2])
    score2 = abs(gripper_x_vector2[2])

    # 더 안전한(점수가 낮은) 자세를 최종 선택
    if score1 <= score2:
        print(f"Choosing original orientation (score: {score1:.2f})")
        return pose1
    else:
        print(f"Choosing 90-deg corrected orientation (score: {score2:.2f})")
        return pose2
    


def apply_rotation_manually(start_pose, zyz_delta):
    """
    개선된 회전 적용 함수 - 더 안정적인 ZYZ 변환
    """
    try:
        # 1. 시작 자세의 오일러 각을 회전 객체로 변환
        start_euler = start_pose[3:]
        r_start = R.from_euler('zyz', start_euler, degrees=True)

        # 2. 적용할 회전 변화량을 회전 객체로 변환
        r_delta = R.from_euler('zyz', zyz_delta, degrees=True)

        # 3. 두 회전을 곱하여 최종 회전 객체를 계산
        # r_final = r_start * r_delta
        r_final = r_delta * r_start  # 변경된 부분

        # 4. 최종 회전 객체를 다시 ZYZ 오일러 각으로 변환
        final_euler = r_final.as_euler('zyz', degrees=True)
        
        # 5. 각도 정규화 (-180 ~ 180도)
        # final_euler = [(angle + 180) % 360 - 180 for angle in final_euler]

        # 6. 원래의 위치(x, y, z)와 새로운 오일러 각을 합쳐 최종 자세 반환
        final_pose = start_pose[:3] + final_euler.tolist()
        
        return final_pose
        
    except Exception as e:
        print(f"Rotation application error: {e}")
        return start_pose  # 오류 시 원래 자세 반환


def correct_colliding_pose(target_pose):
    """
    주어진 목표 자세가 충돌을 유발하는지 확인하고,
    문제가 있다면 툴 롤(Rz')을 90도로 강제 수정하여 반환합니다.
    """
    # is_gripper_collision_expected 함수는 이전에 정의한 것을 그대로 사용합니다.
    if is_gripper_collision_expected(target_pose):
        print("Collision pose detected. Forcing a 90-degree tool roll.")
        
        corrected_pose = target_pose.copy()
        
        # 마지막 Z'축 회전값을 90도로 강제 설정하여 '손날'을 눕힘
        corrected_pose[5] = 90.0
        
        return corrected_pose
    else:
        # 충돌 조건이 아니면 원래 자세 그대로 반환
        return target_pose

def is_gripper_collision_expected(pose, min_rotation_threshold=10.0):
    rx, ry, rz = pose[3], pose[4], pose[5]
    
    # 1. 먼저 최소 회전인지 확인
    # Z축 회전들이 거의 없고 Y축만 90도 근처면 단순한 뒤집기
    # if (abs(rx) < min_rotation_threshold and 
    #     abs(rz) < min_rotation_threshold and 
    #     abs(abs(ry) - 90) < 15):
    #     print(f"단순 뒤집기 감지 (Y={ry:.1f}°) - 충돌 위험 없음")
    #     return False
    
    # 2. 기존 충돌 감지 로직
    is_lying_down = abs(ry - 90) < 15 or abs(ry + 90) < 15
    is_roll_problematic = (abs(rz) < 15 or abs(rz - 180) < 15 or abs(rz + 180) < 15)
    
    if is_lying_down and is_roll_problematic:
        print(f"충돌 위험 감지: Y={ry:.1f}°, Z={rz:.1f}°")
        return True
    
    print(f"안전한 자세: Y={ry:.1f}°, Z={rz:.1f}°")
    return False


# 순응제어 켜기
def on():
    print("Starting force ctrl")
    task_compliance_ctrl(stx=[500, 500, 500, 100, 100, 100])
    wait(0.5)
    set_desired_force(fd=[0, 0, -15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)

# 순응제어 끄기
def off():
    print("Starting release_force")
    release_force()
    wait(0.5)
    release_compliance_ctrl()


class RobotController(Node):
    def __init__(self):
        super().__init__("somacube")
        self.init_robot()
        self.depth_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        while not self.depth_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for depth position service...")
        self.depth_request = SrvDepthPosition.Request()
        # self.robot_control()

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (그리퍼 → 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self, input_data, peice_id, roation, pos):
        user_input = input_data
        peice_id  += 1
        if user_input.lower() == "q":
            self.get_logger().info("Quit the program...")
            sys.exit()

        if user_input:
            # try:
            #     user_input_int = int(user_input)
            #     user_input = tool_dict.get(user_input_int, user_input)
            # except ValueError:
            #     pass  # 변환 불가능하면 원래 문자열 유지
            self.depth_request.target = str(peice_id)
            self.get_logger().info("call depth position service with yolo")
            depth_future = self.depth_client.call_async(self.depth_request)
            rclpy.spin_until_future_complete(self, depth_future)

            if depth_future.result():
                result = depth_future.result().depth_position.tolist()
                self.get_logger().info(f"Received depth position: {result}")
                if sum(result) == 0:
                    print("No target position")
                    return

                gripper2cam_path = os.path.join(
                    PACKAGE_PATH, "resource", "T_gripper2camera.npy"
                )
                robot_posx = get_current_posx()[0]
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

                if td_coord[2] and sum(td_coord) != 0:
                    td_coord[2] += -5  # DEPTH_OFFSET
                    td_coord[2] = max(td_coord[2], 2)  # MIN_DEPTH: float = 2.0

                target_pos = list(td_coord[:3]) + robot_posx[3:]

                self.get_logger().info(f"target position: {target_pos}")
                self.somacube_target(target_pos, roation, pos)
                self.init_robot()
        self.init_robot()
        


    def init_robot(self):
        # JReady = [0, 0, 90, 0, 90, 0]
        JReady = [-14.74, 6.47, 57.94, -0.03, 115.59, -14.74]
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()


    def somacube_target(self, target_pos, rotation, pos):
        """
        수정된 소마큐브 타겟 함수
        """
        
        RE_GRAB_POS = [640, -10.82, 250]
        RE_GRAB_POS_UP = [640, -10.82, 300.00]
        
        print(f"Target position: {target_pos}")
        print(f"Rotation: {rotation}")
        
        # 물체 잡기
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.close_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        mwait()
        print(f"그리퍼 넓이 {gripper.get_width()}")
        while gripper.get_status()[0]:
            time.sleep(0.5)
        
        # 들어올리기
        target_pos_up = up_pos(target_pos, 2, 300)
        movel(target_pos_up, vel=VELOCITY, acc=ACC)

        current_pose, _ = get_current_posx()
        re_pos = RE_GRAB_POS_UP + list(current_pose[3:])
        movel(re_pos, vel=VELOCITY, acc=ACC)
        
        # 회전 적용
        target_pos_rotation = apply_rotation_manually(re_pos, rotation)
        print(f"회전 적용된 목표 자세: {target_pos_rotation}")
        
        # 개선된 충돌 감지 사용
        if is_gripper_collision_expected(target_pos_rotation):
            print("⚠️ 충돌 위험 감지 - 재그랩 로직 적용")
            
            # 특이점 처리
            ry_val = target_pos_rotation[4]
            if abs(ry_val - 90.0) < 0.1 or abs(ry_val + 90.0) < 0.1:
                print(f"Singularity detected at Ry={ry_val:.2f}°. Adjusting to avoid it.")
                target_pos_rotation[4] = 89.9

            # 안전한 조인트 솔루션 찾기
            best_joints = select_safe_joint_solution(target_pos_rotation, "closest")
            if best_joints is not None:
                print(f"재그랩용 조인트: {best_joints}")
                movej(list(best_joints), acc=ACC, vel=VELOCITY)
                wait(0.1)
                
                # 재그랩 로직
                current_j = get_current_posj()
                current_j[5] += 90  # 조인트 6번에 90도 더함
                movej(current_j, vel=VELOCITY, acc=ACC)
                wait(0.1)
                
                # 물체 놓기
                movel(up_pos(get_current_posx()[0], 2, -50), vel=VELOCITY, acc=ACC)
                mwait()
                gripper.open_gripper()
                while gripper.get_status()[0]:
                    time.sleep(0.5)
                current_x, _ = get_current_posx()
                print(f"current_x = {current_x}")
                ro = apply_rotation_manually(current_x, [0,0,90])
                movel(ro, vel=VELOCITY, acc=ACC)

                # 그대로 회전
                mwait()
                gripper.close_gripper()
                while gripper.get_status()[0]:
                    time.sleep(0.5)
                mwait()
                movel(up_pos(current_x, 2, 150), vel=VELOCITY, acc=ACC)

                # 그리퍼를 바닥으로 향하게 (점은 고정, 자세만 변경)
                current_pose = get_current_posx()[0].copy()
                down_pose = current_pose.copy()

                # 현재 Z축 회전을 유지하면서 X, Y축만 조정
                down_pose[3] = current_pose[3]  # 현재 Z축 회전 유지 (Rx)
                down_pose[4] = 180              # Y축 180도로 뒤집기 (Ry)  
                down_pose[5] = current_pose[5]  # 현재 Z축 회전 유지 (Rz)
                best_joints = select_safe_joint_solution(down_pose, "elbow_up")
                movej(list(best_joints), acc=ACC, vel=VELOCITY)
                # movel(down_pose, vel=VELOCITY, acc=ACC)
                print("그리퍼 바닥 방향으로 세움")
                current_j = get_current_posj()
                current_j[5] -= 90  # 조인트 6번에 90도 더함
                movej(current_j, vel=VELOCITY, acc=ACC)
                wait(0.1)

                # 정리
                wait(0.5)
                current_x, _ = get_current_posx()
                home_posx = fkin(HOME, DR_BASE) # posx
                grapping_pos = list(home_posx)[:3] + list(current_x)[3:]
                print("정답으로 출발~")
                self.go_to_answer(grapping_pos, pos)

                
            else:
                print("재그랩 조인트 솔루션 없음 - 직접 이동 시도")
                movel(target_pos_rotation, vel=20, acc=20)
                movel(up_pos(get_current_posx()[0], 2, -55), vel=VELOCITY, acc=ACC)
                gripper.open_gripper()
                mwait()
                while gripper.get_status()[0]:
                    time.sleep(0.5)
        else:
            print("✅ 안전한 자세 - 일반 처리")
            
            # 안전한 조인트 솔루션으로 이동
            best_joints = select_safe_joint_solution(target_pos_rotation, "elbow_down")
            if best_joints is not None:
                print(f"선택된 조인트: {best_joints}")
                movej(list(best_joints), acc=ACC, vel=VELOCITY)
            else:
                print("조인트 솔루션 없음 - 직접 이동")
                movel(target_pos_rotation, vel=30, acc=30)
            
            # 물체 놓기
            print("물체 배치")
            # 정리
            wait(0.5)
            sol = get_current_solution_space()
            current_x_block, _ = get_current_posx()
            movejx(up_pos(current_x_block, 2, -40), vel=VELOCITY, acc=ACC, sol=sol)
            gripper.open_gripper()
            mwait()
            while gripper.get_status()[0]:
                time.sleep(0.5)

            # 정리
            wait(1.0)  # 더 긴 대기
            mwait()    # 이전 동작 완료 확실히 대기
            sol = get_current_solution_space()
            current_x, _ = get_current_posx()
            movejx(up_pos(current_x, 2, 100), vel=VELOCITY, acc=ACC, sol=sol)
            wait(0.5)
            # 다시 세우기 (필요한 경우)
            print("다시 세우기 시도")

            wait(0.5)
            current_x, _ = get_current_posx()
            grapping_pos = current_x_block[:3] + target_pos[3:]
            # amovej(HOME, vel=VELOCITY, acc=ACC, radius = 10)
            wait(0.5)
            best_joints = select_safe_joint_solution(grapping_pos, "closest")
            if best_joints is not None:
                movej(list(best_joints), acc=ACC, vel=VELOCITY)
                wait(0.5)
                current_x, _ = get_current_posx()
                # movel(up_pos(current_x, 2, -50), vel=VELOCITY, acc=ACC)
                mwait()
                gripper.close_gripper()
                while gripper.get_status()[0]:
                    time.sleep(0.5)
        
            # 정리
            wait(0.5)
            current_x, _ = get_current_posx()
            home_posx = fkin(HOME, DR_BASE) # posx
            grapping_pos = list(home_posx)[:3] + list(current_x)[3:]

            print("정답으로 출발~")
            self.go_to_answer(grapping_pos,pos)
    
    def go_to_answer(self, home_pos ,pos):
        current_j = get_current_posj()
        # movej(current_j, vel=VELOCITY, acc=ACC)
        wait(0.1)
        movej(HOME, acc=ACC, vel=VELOCITY)
        after_home = HOME.copy()
        after_home[5] = current_j[5]
        movej(after_home, acc=ACC, vel=VELOCITY)
        # best_joints = select_safe_joint_solution(home_pos, "elbow_up")
        # movej(list(best_joints), acc=ACC, vel=VELOCITY)
        # # wait(0.5)
        # # sol = get_current_solution_space()
        # # movejx(home_pos, vel=VELOCITY, acc=ACC, sol=sol)
        wait(0.5)
        sol = get_current_solution_space()
        current_x, _ = get_current_posx()

        mwait()
        width = gripper.get_width()
        print(f"그리퍼 넓이 {width}")
        while gripper.get_status()[0]:
            time.sleep(0.5)
        
        end_point = ANSWER_POINT + list(current_x[3:])
        for i in range(len(pos)):
            if i == 0:
                end_point[i] += 25*pos[i]
            elif i == 1:
                end_point[i] += 25*pos[i]
        
        # 일단 바꿀 건데 왼 쪽 기준 두칸 일 때
        if  50 <= width <= 75:
            end_point[0] += 12.5
        # 세잔짜리 일때 왼 쪽 기준
        elif 75 <= width <= 91:
            end_point[0] += 25
        
        movejx(end_point, acc=ACC, vel=VELOCITY, sol=sol)
    # 순응제어 및 힘제어 설정
        on()
        while not check_force_condition(DR_AXIS_Z, max=FORCE_VALUE):
            print("Waiting for an external force greater than 5 ")
            wait(0.5)
        off()
        mwait()
        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)
        




######################################################################
# 1. 소마 큐브 조각 및 회전 정의
######################################################################
BASE_PIECES = {
    0: np.array([[0,0,0], [1,0,0], [0,1,0]]), # V 조각
    1: np.array([[0,0,0], [1,0,0], [2,0,0], [2,1,0]]), # L 조각
    2: np.array([[0,0,0], [1,0,0], [2,0,0], [1,1,0]]), # T 조각
    3: np.array([[0,0,0], [1,0,0], [1,1,0], [2,1,0]]), # Z 조각
    4: np.array([[0,0,0], [0,1,0], [1,1,0], [1,1,1]]), # A 조각 (오른손)
    5: np.array([[0,0,0], [1,0,0], [1,1,0], [1,1,1]]), # B 조각 (왼손)
    6: np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), # P 조각
}


def get_all_rotations():
    """ 
    각 조각의 모든 고유한 3D 회전 형태를 미리 계산 (버그 수정 버전)
    """
    all_rotations = {}
    
    rotation_matrices = [
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), # x축
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), # y축
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # z축
    ]

    for i, piece in BASE_PIECES.items():
        seen_normalized_tuples = set()
        unique_orientations_np = []
        queue = [piece]
        
        p_norm = piece - piece.min(axis=0)
        p_tuple = tuple(sorted(map(tuple, p_norm)))
        seen_normalized_tuples.add(p_tuple)
        unique_orientations_np.append(piece)

        head = 0
        while head < len(queue):
            current_piece = queue[head]
            head += 1
            for rot_matrix in rotation_matrices:
                new_p = np.dot(current_piece, rot_matrix)
                new_p_normalized = new_p - new_p.min(axis=0)
                new_p_tuple = tuple(sorted(map(tuple, new_p_normalized)))
                if new_p_tuple not in seen_normalized_tuples:
                    seen_normalized_tuples.add(new_p_tuple)
                    unique_orientations_np.append(new_p)
                    queue.append(new_p)

        final_rotations = []
        for p in unique_orientations_np:
            p_final = p - p.min(axis=0)
            final_rotations.append(p_final)
            
        all_rotations[i] = final_rotations
    return all_rotations

ALL_PIECE_ORIENTATIONS = get_all_rotations()

######################################################################
# 2. 행동 매핑 시스템
######################################################################
class ActionMapper:
    def __init__(self):
        self.action_to_index = {}; self.index_to_action = {}
        index = 0
        for piece_id in range(7):
            for orient_idx in range(len(ALL_PIECE_ORIENTATIONS[piece_id])):
                for x in range(3):
                    for y in range(3):
                        for z in range(3):
                            action = (piece_id, orient_idx, (x, y, z))
                            self.action_to_index[action] = index
                            self.index_to_action[index] = action
                            index += 1
        self.total_actions = index
    def action_to_idx(self, action): return self.action_to_index.get(action, -1)
    def idx_to_action(self, idx): return self.index_to_action.get(idx, None)

######################################################################
# 3. 강화학습 환경
######################################################################
class SomaCubeEnv:
    def __init__(self, action_mapper):
        self.grid_shape = (3, 3, 3); self.action_mapper = action_mapper; self.reset()
    def reset(self):
        self.grid = np.zeros(self.grid_shape, dtype=int); self.pieces_to_place = random.sample(list(range(7)), 7); self.current_piece_idx = self.pieces_to_place.pop(0); self.done = False; return self._get_state()
    def _get_state(self):
        state = np.zeros(27 + 7); state[:27] = self.grid.flatten(); state[27 + self.current_piece_idx] = 1; return state
    def _is_valid_placement(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            if not (0 <= abs_x < 3 and 0 <= abs_y < 3 and 0 <= abs_z < 3) or self.grid[abs_x, abs_y, abs_z] != 0: return False
        return True
    def _is_supported(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_z = position[2] + z
            if abs_z == 0: return True
            if abs_z > 0 and self.grid[position[0] + x, position[1] + y, abs_z - 1] != 0: return True
        return False
    def get_possible_actions(self):
        possible_action_indices = []
        orientations = ALL_PIECE_ORIENTATIONS[self.current_piece_idx]
        for orient_idx, piece_coords in enumerate(orientations):
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        if self._is_valid_placement(piece_coords, (x, y, z)) and self._is_supported(piece_coords, (x, y, z)):
                            action = (self.current_piece_idx, orient_idx, (x, y, z))
                            action_idx = self.action_mapper.action_to_idx(action)
                            if action_idx != -1: possible_action_indices.append(action_idx)
        return possible_action_indices
    def step(self, action_idx):
        action = self.action_mapper.idx_to_action(action_idx)
        if action is None or action[0] != self.current_piece_idx: return self._get_state(), -5.0, True, {"error": "Invalid action"}
        piece_id, orient_idx, position = action
        piece_coords = ALL_PIECE_ORIENTATIONS[self.current_piece_idx][orient_idx]
        if not self._is_valid_placement(piece_coords, position) or not self._is_supported(piece_coords, position): return self._get_state(), -5.0, True, {"error": "Invalid placement"}
        for x, y, z in piece_coords: self.grid[position[0] + x, position[1] + y, position[2] + z] = self.current_piece_idx + 1
        if not self.pieces_to_place: return self._get_state(), 100.0, True, {"success": True}
        self.current_piece_idx = self.pieces_to_place.pop(0)
        if not self.get_possible_actions(): return self._get_state(), -10.0, True, {"error": "Stuck"}
        return self._get_state(), 1.0, False, {}

######################################################################
# 4. DQN 신경망
######################################################################
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__(); self.network = nn.Sequential(nn.Linear(state_size, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, action_size))
    def forward(self, x): return self.network(x.float())
######################################################################
# 5. 회전 계산 함수 (수정된 버전)
######################################################################


def find_rotation_from_points(p1, p2):
    """ 두 점 집합 간의 최적 회전 행렬을 계산합니다 (Kabsch 알고리즘). """
    if p1.shape != p2.shape:
        # L 조각처럼 큐브 수가 다른 경우, 매칭되는 점이 없으므로 회전 계산 불가
        # 이 코드에서는 조각의 큐브 수가 항상 같으므로 이 경우는 발생하지 않음
        return None

    # 각 점 집합의 중심(centroid) 계산
    c1 = np.mean(p1, axis=0)
    c2 = np.mean(p2, axis=0)
    
    # 각 점들을 중심으로 이동
    p1_centered = p1 - c1
    p2_centered = p2 - c2
    
    # 공분산 행렬 H 계산
    H = p1_centered.T @ p2_centered
    
    # 특이값 분해(SVD)를 사용하여 회전 행렬 U, V 계산
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # 회전 행렬 R 계산
    R_mat = V @ U.T
    
    # 반사(reflection)가 발생했는지 확인 및 수정
    if np.linalg.det(R_mat) < 0:
        V[:, -1] *= -1
        R_mat = V @ U.T
        
    return R_mat
######################################################################
# 6. 테스트 실행 (수정된 버전)
######################################################################


def main():
    
    # --- setup ---
    node = RobotController()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_mapper = ActionMapper()
    env = SomaCubeEnv(action_mapper)
    state_size, action_size = 34, action_mapper.total_actions
    policy_net = DQN(state_size, action_size).to(device)
    
    print(f"🤖 Loading trained model from '{CUBE_MODEL_PATH}'")
    try:
        policy_net.load_state_dict(torch.load(CUBE_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{CUBE_MODEL_PATH}'. Please run training script first."); exit()

    policy_net.eval()
    print("Model loaded successfully. Starting test..."); print("-" * 60)

    ##################테스트 스타트###################
    user_input = input("start: ")
    user_input_int = int(user_input)
    while user_input_int == 1:
        
        test_successes = 0

        state = env.reset()
        done = False; step_count = 0; total_reward = 0; solution_path = []

        while not done and step_count < 100:
            possible_actions = env.get_possible_actions()
            if not possible_actions: break
            with torch.no_grad():
                state_tensor = torch.tensor([state], device=device, dtype=torch.float)
                q_values = policy_net(state_tensor)[0]
                best_action_idx = max(possible_actions, key=lambda idx: q_values[idx].item())
            
            solution_path.append(action_mapper.idx_to_action(best_action_idx))
            state, reward, done, info = env.step(best_action_idx)
            total_reward += reward; step_count += 1
        
        if done and "success" in info:
            test_successes += 1
            print(f"✅ Test : SUCCESS")
            print(f"   Reward: {total_reward:.1f}, Steps: {step_count}")
        else:
            print(f"❌ Test : FAILED")
            print(f"   Reward: {total_reward:.1f}, Steps: {step_count}, Reason: {info.get('error', 'Max steps reached')}")
        
        if solution_path:
            print(f"   Placement Path Attempted:")
            for i, step in enumerate(solution_path):
                if step is None:
                    print(f"   - Step {i+1}: Invalid action occurred.")
                    continue
                
                piece_id, orient_idx, pos = step
                print(f"   - Step {i+1}: Place piece {piece_id} at {pos}.")

                base_coords = BASE_PIECES[piece_id]
                rotated_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                
                # 회전 객체 계산
                # rotation = calculate_rotation(base_coords, rotated_coords)

                base_sorted = base_coords[np.lexsort(base_coords.T)]
                rotated_sorted = rotated_coords[np.lexsort(rotated_coords.T)]
                rotation_matrix = find_rotation_from_points(base_sorted, rotated_sorted)
                if rotation_matrix is not None:
                    # Scipy를 사용하여 다양한 회전 표현으로 변환
                    r = R.from_matrix(rotation_matrix)
                    
                    # 축-각도 표현
                    rot_vec = r.as_rotvec()
                    angle_rad = np.linalg.norm(rot_vec)
                    if angle_rad > 1e-6:
                        axis = rot_vec / angle_rad
                        angle_deg = np.rad2deg(angle_rad)
                        print(f"🔄 축-각도: 축 {np.round(axis, 2)}을 중심으로 {angle_deg:.1f}° 회전")
                    else:
                        print("🔄 축-각도: 회전 없음 (0°)")

                    # 오일러 각 표현 (Z-Y-X 순서, 로보틱스에서 흔히 사용)
                    euler_angles = r.as_euler('zyz', degrees=True)
                    print(f"📐 오일러 각 (ZYZ 순): Z축({euler_angles[0]:.1f}°), Y축({euler_angles[1]:.1f}°), Z축({euler_angles[2]:.1f}°) 순으로 회전")
                    node.robot_control(user_input, piece_id, euler_angles, pos)
                else:
                    print("회전 정보를 계산할 수 없습니다.")
                

        
            print("-" * 60)


    rclpy.shutdown()
    node.destroy_node()
    
if __name__ == '__main__':
    main()
