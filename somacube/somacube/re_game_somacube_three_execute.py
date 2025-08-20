import torch
import torch.nn as nn
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import random
import time
from od_msg.srv import SrvDepthPosition
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import DR_init
from somacube.onrobot import RG
import sys
from collections import deque

# ===== 모델 경로 설정 =====
PACKAGE_NAME = "somacube"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
CUBE_MODEL_FILENAME = "robot_friendly_soma_level_5_20250814_151950.pth"
CUBE_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", CUBE_MODEL_FILENAME)


# ===== setup =====

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
dsr_node = rclpy.create_node("somacube_assemble", namespace=ROBOT_ID)
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

########### robot_code ############

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)


re_grap_pos = [624.960, 119.680, -22.780, 68.28, -179.3, 67.66]
HOME = [0, 0, 90, 0, 90, 0]
def up_pos(set_pos, axis, val):
        pos = set_pos.copy()
        pos[axis] += val
        return posx(pos)


def get_true_visual_origin(piece_id, zyz_angles, position):
    """
    ZYZ 각도를 직접 사용하여 원점 위치 계산
    """
    
    if piece_id not in BASE_PIECES:
        return position
    
    # 1. ZYZ 각도로 회전 행렬 생성
    r = R.from_euler('zyz', zyz_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    
    # 2. 조각과 원점 회전
    original_piece = BASE_PIECES[piece_id]
    original_origin = np.array([0, 0, 0])
    
    rotated_piece = np.dot(original_piece, rotation_matrix)
    rotated_origin = np.dot(original_origin, rotation_matrix)
    
    # 3. 정규화
    min_coords = rotated_piece.min(axis=0)
    normalized_origin = rotated_origin - min_coords
    
    # 4. 최종 위치
    final_origin = np.array(position) + normalized_origin
    
    print(f"ZYZ 기반 계산 - Piece {piece_id}:")
    print(f"  ZYZ angles: {zyz_angles}")
    print(f"  Rotation matrix: \n{rotation_matrix}")
    print(f"  Normalized origin: {normalized_origin}")
    print(f"  Final origin: {final_origin}")
    
    return final_origin.astype(int)

def zyz_to_matrix(angles):
    return R.from_euler('zyz', angles, degrees=True).as_matrix()

def bfs_decomposition(target, safe_mats, max_depth=5, tol=1e-6):
    q = deque()
    n = len(safe_mats)
    # 초기 상태 (항등행렬, 빈 경로)
    q.append((np.eye(3), []))
    
    while q:
        mat, path = q.popleft()
        if np.allclose(mat, target, atol=tol):
            return path
        if len(path) >= max_depth:
            continue
        for i, R_s in enumerate(safe_mats):
            new_mat = mat @ R_s
            q.append((new_mat, path + [i]))
    return None


def execute_safe_rotation(re_pos, rotation):
    """최종 간단한 안전 회전 함수"""
    z1, y, z2 = rotation
    
    print(f"🔄 안전 회전: [{z1}, {y}, {z2}]")
    
    # 검증된 안전 패턴들
    safe_patterns = [
        [-90, 90, 90], [90, 90, 90], [0, 0, 90], [0, 0, -90], 
        [90, 0, 0], [-90, 0, 0], [0, 180, 0],[-90, 90, -90], [90,90,-90],
    ]
    
    # 안전 패턴 체크
    for pattern in safe_patterns:
        if (abs(z1 - pattern[0]) < 15 and abs(y - pattern[1]) < 15 and abs(z2 - pattern[2]) < 15):
            print(f"✅ 안전 패턴 - 직접 실행")
            target_pose = apply_rotation_manually(re_pos, rotation)
            joints = select_safe_joint_solution(target_pose, "closest")
            
            if joints is not None and not is_problematic_solution(joints):
                movej(list(joints), vel=VELOCITY, acc=ACC)
            else:
                # 높이 올려서 재시도
                offset_pose = re_pos[:]
                offset_pose[2] += 100
                movel(offset_pose, vel=VELOCITY, acc=ACC)
                target_pose = apply_rotation_manually(offset_pose, rotation)
                joints = select_safe_joint_solution(target_pose, "closest")
                if joints is not None:
                    movej(list(joints), vel=VELOCITY, acc=ACC)
                else:
                    movel(target_pose, vel=15, acc=15)
            
            # 최종 배치하고 끝
            movel(up_pos(get_current_posx()[0], 2, -100), vel=VELOCITY, acc=ACC)
            gripper.open_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            movel(up_pos(get_current_posx()[0], 2, 100), vel=VELOCITY, acc=ACC)
            current_x, _ = get_current_posx()
            intermediate_pose1 = current_x[:]
            intermediate_pose1[4] = 150  # Y축 먼저 150도로
            movel(intermediate_pose1, vel=30, acc=30)
            wait(0.1)
            
            intermediate_pose2 = current_x[:]  
            intermediate_pose2[4] = 180  # Y축 180도로 완성
            movel(intermediate_pose2, vel=30, acc=30)
            wait(0.5)
            movel(up_pos(get_current_posx()[0], 2, -50), vel=VELOCITY, acc=ACC)
            gripper.close_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            return get_current_posx()[0]
    
    # 위험 패턴 분해
    print(f"⚠️ 위험 패턴 - 분해 실행")
    
    
    current_pose = re_pos[:]

    ## 행렬 분해
    if rotation not in safe_patterns:
        safe_mats = [zyz_to_matrix(p) for p in safe_patterns]
        target = zyz_to_matrix(rotation)
        decomp = bfs_decomposition(target, safe_mats, max_depth=5)
        print(f"📋 분해 단계: {len(decomp)}")
    ##
    
    # 각 단계 실행
    for i in range(len(decomp)):
        print(f"📍 {i+1}/{len(decomp)}: {safe_patterns[decomp[i]]}")
        
        target_pose = apply_rotation_manually(current_pose, safe_patterns[decomp[i]])
        joints = select_safe_joint_solution(target_pose, "closest")
        
        if joints is not None and not is_problematic_solution(joints):
            movej(list(joints), vel=VELOCITY, acc=ACC)
        else:
            # 높이 올려서 재시도
            offset_pose = current_pose[:]
            offset_pose[2] += 100
            movel(offset_pose, vel=VELOCITY, acc=ACC)
            target_pose = apply_rotation_manually(offset_pose, safe_patterns[decomp[i]])
            joints = select_safe_joint_solution(target_pose, "closest")
            if joints is not None:
                movej(list(joints), vel=VELOCITY, acc=ACC)
            else:
                movel(target_pose, vel=10, acc=10)
        
        wait(0.3)
        
        # 마지막이 아니면 재그랩
        if i < len(decomp) - 1:
            print(f"   🔄 재그랩")
            movel(up_pos(get_current_posx()[0], 2, -50), vel=VELOCITY, acc=ACC)
            gripper.open_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            
            movel(up_pos(get_current_posx()[0], 2, -100), vel=VELOCITY, acc=ACC)
            current_x, _ = get_current_posx()
            intermediate_pose1 = current_x[:]
            intermediate_pose1[4] = 150  # Y축 먼저 150도로
            movel(intermediate_pose1, vel=30, acc=30)
            wait(0.1)

            intermediate_pose2 = current_x[:]  
            intermediate_pose2[3] = 0
            intermediate_pose2[4] = 180  # Y축 180도로 완성
            intermediate_pose2[5] = 0
            movel(intermediate_pose2, vel=30, acc=30)
            wait(0.5)

            movel(up_pos(get_current_posx()[0], 2, -100), vel=VELOCITY, acc=ACC)
            gripper.close_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            movel(up_pos(get_current_posx()[0], 2, 150), vel=VELOCITY, acc=ACC)
        
        current_pose = get_current_posx()[0]
    
    # 최종 배치
    print("🎯 최종 배치")
    movel(up_pos(current_pose, 2, -100), vel=VELOCITY, acc=ACC)
    gripper.open_gripper()
    while gripper.get_status()[0]: time.sleep(0.5)
    
    current_x, _ = get_current_posx()
    # safe_pose = [current_x[0], current_x[1], current_x[2] + 100, 0, 180, 0]
    # movel(safe_pose, vel=VELOCITY, acc=ACC)
    # 단계적 자세 변경 (90도 단위)
    intermediate_pose1 = current_x[:]
    intermediate_pose1[4] = 150  # Y축 먼저 150도로
    movel(intermediate_pose1, vel=30, acc=30)
    wait(0.1)
    
    intermediate_pose2 = current_x[:]  
    intermediate_pose2[3] = 0
    intermediate_pose2[4] = 180  # Y축 180도로 완성
    intermediate_pose2[5] = 0
    movel(intermediate_pose2, vel=30, acc=30)
    wait(0.5)
    movel(up_pos(get_current_posx()[0], 2, -50), vel=VELOCITY, acc=ACC)
    gripper.close_gripper()
    while gripper.get_status()[0]: time.sleep(0.5)
    
    print("✅ 회전 완료")
    return get_current_posx()[0]

def is_any_rotation_needed(rotation):
    """회전이 필요한지 간단히 체크"""
    z1, y, z2 = rotation
    return abs(z1) > 5 or abs(y) > 5 or abs(z2) > 5


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
        r_final = r_start * r_delta


        # 4. 최종 회전 객체를 다시 ZYZ 오일러 각으로 변환
        final_euler = r_final.as_euler('zyz', degrees=True)
        
        # 5. 각도 정규화 (-180 ~ 180도)
        final_euler = [(angle + 180) % 360 - 180 for angle in final_euler]

        # 6. 원래의 위치(x, y, z)와 새로운 오일러 각을 합쳐 최종 자세 반환
        # final_pose = start_pose[:3] + final_euler.tolist()
        final_pose = start_pose[:3] + final_euler
        
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
    #충돌감지
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

    def robot_control(self, input_data, piece_id, rotaion, pos):
        user_input = str(input_data)
        piece_id  += 1
        if user_input.lower() == "q":
            self.get_logger().info("Quit the program...")
            sys.exit()

        if user_input:
            # try:
            #     user_input_int = int(user_input)
            #     user_input = tool_dict.get(user_input_int, user_input)
            # except ValueError:
            #     pass  # 변환 불가능하면 원래 문자열 유지
            self.depth_request.target = str(piece_id)
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
                self.somacube_target(target_pos, rotaion, pos)
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
        RE_GRAB_POS_UP = [437.35, -10.82, 300.00]
        
        
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
        
        print("re grap pos 로")
        # if is_any_rotation_needed(rotation):
        final_pose = execute_safe_rotation(re_pos, rotation)
        self.go_to_answer(final_pose, pos)

    
    def go_to_answer(self, final_pos ,pos):
        movel(up_pos(get_current_posx()[0], 2, 100), vel=VELOCITY, acc=ACC)

        sol = get_current_solution_space()
        end_point = ANSWER_POINT + list(final_pos[3:])
        for i in range(len(pos)):
            if i == 0:
                end_point[i] += 25*pos[i]
            elif i == 1:
                end_point[i] += 25*pos[i]
        
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
        


BASE_PIECES = {
    0: np.array([[0,0,0], [1,0,0], [0,1,0]]), # V 조각
    1: np.array([[0,0,0], [1,0,0], [2,0,0], [2,1,0]]), # L 조각
    2: np.array([[0,0,0], [1,0,0], [2,0,0], [1,1,0]]), # T 조각
    3: np.array([[0,0,0], [1,0,0], [1,1,0], [2,1,0]]), # Z 조각
    4: np.array([[0,0,0], [0,1,0], [1,1,0], [1,1,1]]), # A 조각
    5: np.array([[0,0,0], [1,0,0], [1,1,0], [1,1,1]]), # B 조각
    6: np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), # P 조각
}

PIECE_NAMES = {0: "V", 1: "L", 2: "T", 3: "Z", 4: "A", 5: "B", 6: "P"}

# ===== 회전 시스템 =====
def get_all_rotations_with_matrices():
    all_rotations = {}
    rotation_matrices_info = {}
    
    rotation_matrices = []
    matrix_descriptions = []
    
    # 기본 회전들
    for axis in ['x', 'y', 'z']:
        for angle in [0, 90, 180, 270]:
            if axis == 'x':
                if angle == 0:
                    rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    desc = "identity"
                elif angle == 90:
                    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    desc = "x90"
                elif angle == 180:
                    rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    desc = "x180"
                else:
                    rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    desc = "x270"
            elif axis == 'y':
                if angle == 0:
                    rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    desc = "identity"
                elif angle == 90:
                    rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                    desc = "y90"
                elif angle == 180:
                    rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    desc = "y180"
                else:
                    rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    desc = "y270"
            else:  # z
                if angle == 0:
                    rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    desc = "identity"
                elif angle == 90:
                    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                    desc = "z90"
                elif angle == 180:
                    rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    desc = "z180"
                else:
                    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                    desc = "z270"
            
            if desc != "identity" or len(rotation_matrices) == 0:
                rotation_matrices.append(rot)
                matrix_descriptions.append(desc)
    
    # 추가 회전들
    additional_rotations = [
        (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), "x90_z90"),
        (np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), "x90_z270"),
        (np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), "x270_z90"),
        (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]), "x270_z270"),
        (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), "y90_x90"),
        (np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), "y90_x270"),
        (np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]), "y270_x90"),
        (np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]), "y270_x270"),
    ]
    
    for rot, desc in additional_rotations:
        rotation_matrices.append(rot)
        matrix_descriptions.append(desc)
    
    # 각 조각별로 회전 계산
    for piece_id, piece in BASE_PIECES.items():
        seen_signatures = set()
        unique_orientations = []
        used_matrices = []
        used_descriptions = []
        
        for i, rot_matrix in enumerate(rotation_matrices):
            new_p = np.dot(piece, rot_matrix)
            new_p_normalized = new_p - new_p.min(axis=0)
            coords_set = frozenset(tuple(coord) for coord in new_p_normalized)
            
            if coords_set not in seen_signatures:
                seen_signatures.add(coords_set)
                p_final = np.round(new_p_normalized).astype(int)
                unique_orientations.append(p_final)
                used_matrices.append(rot_matrix)
                used_descriptions.append(matrix_descriptions[i])
        
        all_rotations[piece_id] = unique_orientations
        rotation_matrices_info[piece_id] = {
            'matrices': used_matrices,
            'descriptions': used_descriptions
        }
    
    return all_rotations, rotation_matrices_info

def get_zyz_angles(piece_id, orientation_index):
    if piece_id not in ROTATION_MATRICES_INFO:
        return [0, 0, 0], "error"
    
    matrices_info = ROTATION_MATRICES_INFO[piece_id]
    if orientation_index >= len(matrices_info['matrices']):
        return [0, 0, 0], "error"
    
    rotation_matrix = matrices_info['matrices'][orientation_index]
    description = matrices_info['descriptions'][orientation_index]
    
    try:
        U, s, Vt = np.linalg.svd(rotation_matrix)
        corrected_matrix = U @ Vt
        
        if np.linalg.det(corrected_matrix) < 0:
            Vt[-1, :] *= -1
            corrected_matrix = U @ Vt
        
        r = R.from_matrix(corrected_matrix)
        zyz_angles = r.as_euler('ZYZ', degrees=True)
        
        return zyz_angles.tolist(), description
        
    except Exception as e:
        return [0, 0, 0], "error"

# 전역 변수로 회전 정보 초기화
ALL_PIECE_ORIENTATIONS, ROTATION_MATRICES_INFO = get_all_rotations_with_matrices()

class SomaCubeEnv:
    def __init__(self, max_pieces=2):
        self.max_pieces = max_pieces
        self.reset()
    
    def reset(self):
        self.grid = np.zeros((3, 3, 3), dtype=int)
        all_pieces = list(range(7))
        self.pieces_to_place = random.sample(all_pieces, self.max_pieces)
        self.current_piece_idx = 0
        self.placed_pieces = []
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        state = np.zeros(36)
        state[:27] = self.grid.flatten()
        
        if self.current_piece_idx < len(self.pieces_to_place):
            current_piece = self.pieces_to_place[self.current_piece_idx]
            state[27 + current_piece] = 1
        
        state[34] = len(self.placed_pieces) / self.max_pieces
        state[35] = self.current_piece_idx / self.max_pieces
        return state
    
    def _has_clear_vertical_path(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            if not (0 <= abs_x < 3 and 0 <= abs_y < 3 and 0 <= abs_z < 3):
                continue
            for check_z in range(abs_z + 1, 3):
                if self.grid[abs_x, abs_y, check_z] != 0:
                    return False
        return True
    
    def _is_valid_placement(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            if not (0 <= abs_x < 3 and 0 <= abs_y < 3 and 0 <= abs_z < 3):
                return False
            if self.grid[abs_x, abs_y, abs_z] != 0:
                return False
        return True
    
    def _is_supported(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_z = position[2] + z
            if abs_z == 0:
                return True
            if abs_z > 0:
                abs_x, abs_y = position[0] + x, position[1] + y
                if 0 <= abs_x < 3 and 0 <= abs_y < 3:
                    if self.grid[abs_x, abs_y, abs_z - 1] != 0:
                        return True
        return False
    
    def get_possible_actions(self):
        if self.current_piece_idx >= len(self.pieces_to_place):
            return []
        
        possible_actions = []
        piece_id = self.pieces_to_place[self.current_piece_idx]
        orientations = ALL_PIECE_ORIENTATIONS[piece_id]
        
        for orient_idx, piece_coords in enumerate(orientations):
            for z in range(3):
                for x in range(3):
                    for y in range(3):
                        position = (x, y, z)
                        if (self._is_valid_placement(piece_coords, position) and 
                            self._is_supported(piece_coords, position) and
                            self._has_clear_vertical_path(piece_coords, position)):
                            action = (piece_id, orient_idx, position)
                            possible_actions.append(action)
        
        # 바닥층 우선 정렬
        def sort_key(action):
            piece_id, orient_idx, position = action
            piece_coords = orientations[orient_idx]
            min_z = min(position[2] + z for x, y, z in piece_coords)
            ground_blocks = sum(1 for x, y, z in piece_coords if position[2] + z == 0)
            return (min_z, -ground_blocks, position[2])
        
        possible_actions.sort(key=sort_key)
        return possible_actions
    
    def step(self, action):
        piece_id, orient_idx, position = action
        piece_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
        
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            self.grid[abs_x, abs_y, abs_z] = piece_id + 1
        
        self.placed_pieces.append((piece_id, orient_idx, position))
        self.current_piece_idx += 1
        
        if self.current_piece_idx >= len(self.pieces_to_place):
            self.done = True
            return self._get_state(), True, {"success": True}
        
        return self._get_state(), False, {}

# ===== DQN 모델 =====
class HierarchicalDQN(nn.Module):
    def __init__(self, state_size, max_orientations=30, max_positions=27):
        super(HierarchicalDQN, self).__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.orientation_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_orientations)
        )
        
        self.position_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_positions)
        )
    
    def forward(self, x):
        features = self.feature_network(x)
        orientation_q = self.orientation_network(features)
        position_q = self.position_network(features)
        return orientation_q, position_q

# ===== 로봇 테스터 =====
class RobotTester:
    def __init__(self, device='cpu'):
        self.node = RobotController()
        self.device = device
        self.model = None
        self.loaded_level = None
        
        # 지정된 모델 파일 로딩
        self._load_specified_model()
    
    def _load_specified_model(self):
        """지정된 경로의 모델 로딩"""
        if not os.path.exists(CUBE_MODEL_PATH):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {CUBE_MODEL_PATH}")
            print(f"📁 다음 경로에 '{CUBE_MODEL_FILENAME}' 파일을 배치하세요:")
            print(f"   {os.path.dirname(CUBE_MODEL_PATH)}")
            return
        
        try:
            # PyTorch 2.6+ 호환성을 위해 weights_only=False 명시
            checkpoint = torch.load(CUBE_MODEL_PATH, map_location=self.device, weights_only=False)
            
            # 레벨 정보 추출 (없으면 기본값 2)
            self.loaded_level = checkpoint.get('level', 2)
            
            self.model = HierarchicalDQN(36).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ 모델 로딩 완료: {CUBE_MODEL_PATH}")
            print(f"📊 레벨: {self.loaded_level}")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print(f"💡 모델 파일이 올바른 형식인지 확인하세요.")
    
    def _auto_load_model(self):
        """자동으로 최신 모델 로딩 (사용 안 함)"""
        pass
    
    def _find_latest_model(self, level, model_dir="models"):
        """최신 모델 찾기 (사용 안 함)"""
        return None
    
    def _load_model(self, model_path, level):
        """모델 로딩 (사용 안 함)"""
        return False
    
    def _get_valid_actions_for_piece(self, piece_id, env):
        valid_actions = []
        orientations = ALL_PIECE_ORIENTATIONS[piece_id]
        
        for orient_idx, piece_coords in enumerate(orientations):
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        position = (x, y, z)
                        if (env._is_valid_placement(piece_coords, position) and 
                            env._is_supported(piece_coords, position) and
                            env._has_clear_vertical_path(piece_coords, position)):
                            valid_actions.append((orient_idx, x * 9 + y * 3 + z))
        return valid_actions
    
    def _select_action(self, state, piece_id, env):
        if self.model is None:
            return None
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            orientation_q, position_q = self.model(state_tensor)
            
            valid_actions = self._get_valid_actions_for_piece(piece_id, env)
            if not valid_actions:
                return None
            
            best_value = float('-inf')
            best_action = None
            
            for orient_idx, pos_idx in valid_actions:
                if orient_idx < orientation_q.shape[1] and pos_idx < position_q.shape[1]:
                    value = orientation_q[0][orient_idx].item() + position_q[0][pos_idx].item()
                    if value > best_value:
                        best_value = value
                        best_action = (orient_idx, pos_idx)
            
            if best_action is None:
                return None
            
            orient_idx, pos_idx = best_action
        
        x = pos_idx // 9
        y = (pos_idx % 9) // 3
        z = pos_idx % 3
        
        return (piece_id, orient_idx, (x, y, z))
    
    def __del__(self):
        try:
            if hasattr(self, 'node') and self.node:
                self.node.destroy_node()
        except:
            pass
    
    def test_robot_control(self, num_tests=1):
        """로봇 제어 테스트"""
        if self.model is None:
            print("❌ 모델이 로딩되지 않았습니다.")
            return
        
        print(f"🤖 소마큐브 로봇 제어 테스트 시작 (레벨 {self.loaded_level}, {num_tests}회)")
        print("="*60)
        
        env = SomaCubeEnv(max_pieces=self.loaded_level)
        success_count = 0
        
        for test in range(num_tests):
            state = env.reset()
            
            print(f"\n🧩 테스트 {test + 1}:")
            print(f"   사용할 조각들: {[PIECE_NAMES[p] for p in env.pieces_to_place]}")
            
            robot_commands = []
            steps = 0
            
            while not env.done and steps < 20:
                if env.current_piece_idx >= len(env.pieces_to_place):
                    break
                
                piece_id = env.pieces_to_place[env.current_piece_idx]
                action = self._select_action(state, piece_id, env)
                
                if action is None:
                    print(f"   ❌ 단계 {steps + 1}: 가능한 행동이 없음")
                    break
                
                # 행동 실행
                next_state, done, info = env.step(action)
                
                # 로봇 제어 명령 생성
                piece_id, orient_idx, position = action
                piece_name = PIECE_NAMES[piece_id]
                piece_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                zyz_angles, rotation_desc = get_zyz_angles(piece_id, orient_idx)
                
                # 바닥층 여부 확인
                min_z = min(position[2] + z for x, y, z in piece_coords)
                level_info = "바닥층" if min_z == 0 else f"{min_z}층"
                
                command = f"node.robot_control(user_input, {piece_id}, {zyz_angles}, {position})"
                robot_commands.append(command)
                
                print(f"   🔧 단계 {steps + 1}: {piece_name} 조각 → 위치 {position} ({level_info})")
                print(f"       회전: {rotation_desc} → ZYZ{zyz_angles}")
                print(f"       명령: {command}")

                true_origin = get_true_visual_origin(piece_id, zyz_angles, position)
                # true_origin = get_true_visual_origin(piece_id, orient_idx, position)
                print(f"    실제위치: {true_origin}")
                self.node.robot_control(num_tests, piece_id, zyz_angles, true_origin)
                # self.node.robot_control(num_tests, piece_id, [-180, 180, 0], true_origin)
                
                state = next_state
                steps += 1
            
            success = env.done and "success" in info
            if success:
                success_count += 1
            
            status = "✅ 성공" if success else "❌ 실패"
            print(f"   {status} ({steps}단계)")
            
            if success and num_tests == 1:
                print(f"\n🤖 로봇 제어 스크립트:")
                print("="*50)
                for i, cmd in enumerate(robot_commands):
                    print(f"# 단계 {i+1}")
                    print(cmd)
                print("="*50)
        
        print(f"\n📊 최종 결과: {success_count}/{num_tests} 성공 ({success_count/num_tests:.1%})")
# ===== 메인 실행 =====
def main():
    robot_tester = RobotTester()
    
    if robot_tester.model is None:
        print("프로그램을 종료합니다.")
        return
    
    try:
        num_tests = int(input(f"🔢 테스트 횟수를 입력하세요 (기본 1): ") or "1")
        robot_tester.test_robot_control(num_tests)
        
    except KeyboardInterrupt:
        print("\n👋 프로그램 종료")
    except ValueError:
        print("❌ 올바른 숫자를 입력하세요.")
        robot_tester.test_robot_control(1)
    except Exception as e:
        print(f"💥 오류 발생: {e}")
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()