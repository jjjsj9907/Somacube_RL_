import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import time
from datetime import datetime
import pickle
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import DR_init
from somacube.onrobot import RG
import sys
import rclpy
from od_msg.srv import SrvDepthPosition
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor
import threading

# ===== 모델 경로 설정 =====
PACKAGE_NAME = "somacube"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
CUBE_MODEL_FILENAME = "ultimate_soma_final.pt"
CUBE_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", CUBE_MODEL_FILENAME)


# ===== setup =====

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
UP_JOG = [8.81, 5.70, 59.50, -7.02, 90.07, -6.1]
# ANSWER_POINT = [424.850, 78.830, 12.4] # 총 7.5cm 개당 2.5 cm
# ANSWER_POINT = [449.850, 53.830, 100]
# ANSWER_POINT = [437.35, 16.33, 120]
ANSWER_POINT = [437.35, 6.33, 150]

FORCE_VALUE = 10

STOP_FLAG = False
START_FLAG = False


tool_dict = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7"}

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("somacube_assemble", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

executor = MultiThreadedExecutor()
executor.add_node(dsr_node)

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait,\
                           trans, wait, DR_BASE, amovel, amovej, amovejx,\
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


# CUDA 환경 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
warnings.filterwarnings('ignore')

def initialize_gpu_safely():
    """GPU 안전 초기화"""
    try:
        if not torch.cuda.is_available():
            print("⚠️ CUDA 사용 불가, CPU로 진행")
            return torch.device('cpu')
        
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        
        # 테스트
        test_tensor = torch.zeros(1).to(device)
        _ = test_tensor + 1
        print(f"🎯 GPU 초기화 완료: {device}")
        return device
    except Exception as e:
        print(f"⚠️ GPU 초기화 실패: {e}, CPU 사용")
        return torch.device('cpu')

DEVICE = initialize_gpu_safely()



## 기존 방식
def get_true_visual_origin(piece_coords, position):    
    x_list = []
    abs_x_list = []
    abs_y_list = []
    abs_z_list = []
    z_coord = 0
    for x, y, z in piece_coords:
        abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
        if 0 <= abs_x < 3 and 0 <= abs_y < 3 and 0 <= abs_z < 3:
            print(f"x = {x}, y = {y}, z = {z}")
            print(f"")
            print(f"abs_x = {abs_x}, abs_y = {abs_y}, abs_z = {abs_z}")
            abs_x_list.append(abs_x)
            abs_y_list.append(abs_y)
            abs_z_list.append(abs_z)
            x_list.append(abs_x)
            if abs_y_list[0] == abs_y:
                if z_coord < abs_z:
                    z_coord = abs_z
    set_x = set(x_list)
    average_x = sum(set_x) / len(set_x)

    return [abs_x_list[0], abs_y_list[0], abs_z_list[0], average_x, z_coord]

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

def get_x(max_attempts=10):
    import time
    
    for attempt in range(max_attempts):
        try:
            x, _ = get_current_posx()
            if x and len(x) == 6:  # 데이터 유효성 검사
                return x
        except (IndexError, Exception) as e:
            print(f"시도 {attempt + 1} 실패: {e}")
        
        time.sleep(0.1)  # 짧은 지연
    
    print("최대 시도 횟수 초과")
    return None

def execute_safe_rotation(re_pos, rotation):
    """최종 간단한 안전 회전 함수"""
    z1, y, z2 = rotation
    
    print(f"🔄 안전 회전: [{z1}, {y}, {z2}]")
    
    # 검증된 안전 패턴들
    safe_patterns = [
        [-90, 90, 90], [90, 90, 90], [0, 0, 90], [0, 0, -90], 
        [90, 0, 0], [-90, 0, 0],[-90, 90, -90], [90,90,-90],
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
            movel(up_pos(get_x(), 2, -100), vel=VELOCITY, acc=ACC)
            mwait()
            gripper.open_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            movel(up_pos(get_x(), 2, 100), vel=VELOCITY, acc=ACC)
            current_x = get_x()
            intermediate_pose1 = current_x[:]
            intermediate_pose1[4] = 150  # Y축 먼저 150도로
            amovel(intermediate_pose1, vel=30, acc=30, radius=10)
            wait(0.1)
            
            intermediate_pose2 = current_x[:]  
            intermediate_pose2[3] = 0
            intermediate_pose2[4] = 180  # Y축 180도로 완성
            intermediate_pose2[5] = 0
            movel(intermediate_pose2, vel=30, acc=30)
            movel(up_pos(get_x(), 2, -50), vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            return get_x()
    
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
            movel(up_pos(get_x(), 2, -50), vel=VELOCITY, acc=ACC)

            mwait()
            gripper.open_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            
            movel(up_pos(get_x(), 2, 100), vel=VELOCITY, acc=ACC)
            current_x = get_x()
            intermediate_pose1 = current_x[:]
            intermediate_pose1[4] = 150  # Y축 먼저 150도로
            amovel(intermediate_pose1, vel=30, acc=30, radius=10)

            intermediate_pose2 = current_x[:]  
            intermediate_pose2[3] = 0
            intermediate_pose2[4] = 180  # Y축 180도로 완성
            intermediate_pose2[5] = 0
            movel(intermediate_pose2, vel=30, acc=30)

            movel(up_pos(get_x(), 2, -100), vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            while gripper.get_status()[0]: time.sleep(0.5)
            movel(up_pos(get_x(), 2, 150), vel=VELOCITY, acc=ACC)
        current_pose = get_x()
    
    # 최종 배치
    print("🎯 최종 배치")
    movel(up_pos(current_pose, 2, -100), vel=VELOCITY, acc=ACC)
    gripper.open_gripper()
    while gripper.get_status()[0]: time.sleep(0.5)
    current_x = get_x()
    # safe_pose = [current_x[0], current_x[1], current_x[2] + 100, 0, 180, 0]
    # movel(safe_pose, vel=VELOCITY, acc=ACC)
    # 단계적 자세 변경 (90도 단위)
    intermediate_pose1 = current_x[:]
    intermediate_pose1[4] = 150  # Y축 먼저 150도로
    amovel(intermediate_pose1, vel=30, acc=30, radius=10)
    wait(0.1)
    
    intermediate_pose2 = current_x[:]  
    intermediate_pose2[3] = 0
    intermediate_pose2[4] = 180  # Y축 180도로 완성
    intermediate_pose2[5] = 0
    movel(intermediate_pose2, vel=30, acc=30)
    movel(up_pos(get_x(), 2, -50), vel=VELOCITY, acc=ACC)
    gripper.close_gripper()
    while gripper.get_status()[0]: time.sleep(0.5)
    
    print("✅ 회전 완료")
    return get_x()

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
        if STOP_FLAG == True:
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
                robot_posx = get_x()
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

                if td_coord[2] and sum(td_coord) != 0:
                    td_coord[2] += -5  # DEPTH_OFFSET
                    td_coord[2] = max(td_coord[2], 2)  # MIN_DEPTH: float = 2.0

                target_pos = list(td_coord[:3]) + robot_posx[3:]

                self.get_logger().info(f"target position: {target_pos}")
                self.somacube_target(target_pos, rotaion, pos)
                self.init_robot()

        


    def init_robot(self):
        # JReady = [0, 0, 90, 0, 90, 0]
        JReady = [-14.74, 6.47, 57.94, -0.03, 115.59, -14.74]
        movej(JReady, vel=VELOCITY, acc=ACC)
        movel(up_pos(get_x(), 2, -100), vel=VELOCITY, acc=ACC)
        mwait()
        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)



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
        
        
        # 들어올리기
        target_pos_up = up_pos(target_pos, 2, 300)
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        current_pose = get_x()
        re_pos = RE_GRAB_POS_UP + list(current_pose[3:])
        amovel(re_pos, vel=VELOCITY, acc=ACC, radius=10)
        
        print("re grap pos 로")
        # if is_any_rotation_needed(rotation):
        final_pose = execute_safe_rotation(re_pos, rotation)
        self.go_to_answer(final_pose, pos)

    
    def go_to_answer(self, final_pos ,pos):
        movel(up_pos(get_x(), 2, 100), vel=VELOCITY, acc=ACC)
        mwait()
        width_gripper =gripper.get_width() 
        while gripper.get_status()[0]:
            time.sleep(0.5) 
        print(f"그리퍼 넓이 {width_gripper}")

        sol = get_current_solution_space()
        end_point = ANSWER_POINT + list(final_pos[3:])
        amovejx(end_point, acc=ACC, vel=VELOCITY, sol=sol, radius=10)
        end_point_2 = end_point.copy()
        for i in range(len(pos)):
            if i == 0:
                end_point[i] += 25*pos[i]
            elif i == 1:
                end_point[i] += 25*pos[i]
            elif i == 2:
                end_point[i] = pos[i]*25 + 25
        
        if 50 <= width_gripper <= 75:
            if pos[0] < pos[3]:
                end_point[0] += 12.5
            else:
                end_point[0] -= 12.5
        elif 75 <= width_gripper <= 100:
            if pos[0] < pos[3]:
                end_point[0] += 25
            else:
                end_point[0] -= 25
        
        end_point_2[0] = end_point[0]
        end_point_2[1] = end_point[1]
        movel(end_point_2, acc=ACC, vel=VELOCITY)
        end_z = 12.4 + 25 + 25*pos[4]
        end_point_2[2] = end_z
        movel(up_pos(end_point_2 , 2 , 12.5), acc=ACC, vel=VELOCITY)
        
    # 순응제어 및 힘제어 설정
        on()
        while not check_force_condition(DR_AXIS_Z, max=FORCE_VALUE):
            print("Waiting for an external force greater than 5 ")
            wait(0.5)
        off()
        mwait()
        

        gripper.open_gripper_endding(int(width_gripper * 10 + 200))
        while gripper.get_status()[0]:
            time.sleep(0.5)

        movel(up_pos(get_x(), 2, 100), vel=VELOCITY, acc=ACC)


# ===== 기본 정의들 =====
BASE_PIECES = {
    0: np.array([[0,0,0], [1,0,0], [0,1,0]]),  # V (3블록)
    1: np.array([[0,0,0], [1,0,0], [2,0,0], [2,1,0]]),  # L (4블록)
    2: np.array([[0,0,0], [1,0,0], [2,0,0], [1,1,0]]),  # T (4블록)
    3: np.array([[0,0,0], [1,0,0], [1,1,0], [2,1,0]]),  # Z (4블록)
    4: np.array([[0,0,0], [0,1,0], [1,1,0], [1,1,1]]),  # A (4블록)
    5: np.array([[0,0,0], [1,0,0], [1,1,0], [1,1,1]]),  # B (4블록)
    6: np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]),  # P (4블록)
}

PIECE_NAMES = {0: "V", 1: "L", 2: "T", 3: "Z", 4: "A", 5: "B", 6: "P"}

def get_all_rotations_with_matrices():
    """회전 행렬 정보도 함께 저장하는 회전 계산 함수"""
    all_rotations = {}
    rotation_matrices_info = {}
    
    rotation_matrices = []
    matrix_descriptions = []
    
    # X, Y, Z 축 각각에 대해 0, 90, 180, 270도 회전
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
                else:  # 270
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
                else:  # 270
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
                else:  # 270
                    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                    desc = "z270"
            
            if desc != "identity" or len(rotation_matrices) == 0:
                rotation_matrices.append(rot)
                matrix_descriptions.append(desc)
    
    # 추가적인 회전들
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
ALL_PIECE_ORIENTATIONS, ROTATION_MATRICES_INFO = get_all_rotations_with_matrices()

def get_zyz_angles(piece_id, orientation_index):
    """회전 인덱스를 ZYZ 오일러 각으로 변환"""
    if piece_id not in ROTATION_MATRICES_INFO:
        return [0, 0, 0], "error"
    
    matrices_info = ROTATION_MATRICES_INFO[piece_id]
    if orientation_index >= len(matrices_info['matrices']):
        return [0, 0, 0], "error"
    
    rotation_matrix = matrices_info['matrices'][orientation_index]
    description = matrices_info['descriptions'][orientation_index]
    
    try:
        # Kabsch 알고리즘으로 회전 행렬 구하기
        src = BASE_PIECES[piece_id].astype(float)
        dst = ALL_PIECE_ORIENTATIONS[piece_id][orientation_index].astype(float)
        
        src_centered = src - src.mean(axis=0)
        dst_centered = dst - dst.mean(axis=0)
        
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        
        # 반사 방지
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
        
        # ZYZ 오일러각 변환
        rot = R.from_matrix(R_mat)
        euler = rot.as_euler('ZYZ', degrees=True)
        
        return euler.tolist(), description
    except:
        return [0, 0, 0], description

# ===== 적응적 보상 시스템 =====
class AdaptiveRewardCalculator:
    def __init__(self, curriculum_level=1):
        self.curriculum_level = curriculum_level
        self.base_reward = 15.0 + (curriculum_level * 5.0)
        self.connectivity_weight = 4.0
        self.stability_weight = 3.0
        self.progress_weight = 25.0
        self.completion_bonus = 150.0 + (curriculum_level * 50.0)
        self.failure_penalty = -8.0

# ===== 스마트 소마큐브 환경 =====
class SmartSomaCubeEnv:
    def __init__(self, curriculum_level=1, adaptive_difficulty=True):
        self.grid_shape = (3, 3, 3)
        self.curriculum_level = curriculum_level
        self.adaptive_difficulty = adaptive_difficulty
        self.reward_calculator = AdaptiveRewardCalculator(curriculum_level)
        
        self.curriculum_pieces = {
            1: [0, 1, 2],  # V, L, T (3조각, 11블록)
            2: [0, 1, 2, 3, 4],  # V, L, T, Z, A (5조각, 19블록)
            3: list(range(7))  # 모든 조각 (7조각, 27블록)
        }
        
        self.reset()
    
    def reset(self):
        self.grid = np.zeros(self.grid_shape, dtype=np.int8)
        available_pieces = self.curriculum_pieces[self.curriculum_level]
        
        if self.adaptive_difficulty and hasattr(self, 'recent_success_rate'):
            if self.recent_success_rate > 0.8:
                num_pieces = min(len(available_pieces), len(available_pieces))
            elif self.recent_success_rate < 0.2:
                num_pieces = max(2, len(available_pieces) - 1)
            else:
                num_pieces = len(available_pieces)
        else:
            num_pieces = len(available_pieces)
        
        self.pieces_to_place = random.sample(available_pieces, num_pieces)
        self.current_piece_idx = 0
        self.placed_pieces = []
        self.done = False
        self.step_count = 0
        self.max_steps = num_pieces * 50
        
        return self._get_state()
    
    def _get_state(self):
        grid_state = self.grid.flatten().astype(np.float32)
        
        piece_state = np.zeros(7, dtype=np.float32)
        if self.current_piece_idx < len(self.pieces_to_place):
            current_piece = self.pieces_to_place[self.current_piece_idx]
            piece_state[current_piece] = 1.0
        
        progress_state = np.array([
            len(self.placed_pieces) / len(self.pieces_to_place),
            self.current_piece_idx / len(self.pieces_to_place),
            self.step_count / self.max_steps,
            self.curriculum_level / 3
        ], dtype=np.float32)
        
        context_state = np.array([
            np.sum(self.grid > 0) / 27,
            np.sum(self.grid[:, :, 0] > 0) / 9,
            np.sum(self.grid[:, :, 1] > 0) / 9,
            np.sum(self.grid[:, :, 2] > 0) / 9,
        ], dtype=np.float32)
        
        return np.concatenate([grid_state, piece_state, progress_state, context_state])
    
    def get_valid_actions(self, use_smart_filtering=True):
        if self.current_piece_idx >= len(self.pieces_to_place):
            return []
        
        piece_id = self.pieces_to_place[self.current_piece_idx]
        orientations = ALL_PIECE_ORIENTATIONS[piece_id]
        valid_actions = []
        
        for orient_idx, piece_coords in enumerate(orientations):
            for z in range(3):
                for x in range(3):
                    for y in range(3):
                        position = (x, y, z)
                        
                        if self._is_valid_placement(piece_coords, position):
                            valid_actions.append((piece_id, orient_idx, position))
        
        def priority_key(action):
            _, orient_idx, position = action
            piece_coords = orientations[orient_idx]
            min_z = min(position[2] + z for x, y, z in piece_coords)
            ground_blocks = sum(1 for x, y, z in piece_coords if position[2] + z == 0)
            connectivity = self._count_adjacent_blocks(piece_coords, position)
            
            return (min_z, -ground_blocks, -connectivity)
        
        valid_actions.sort(key=priority_key)
        return valid_actions
    
    def _is_valid_placement(self, piece_coords, position):
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            
            if not (0 <= abs_x < 3 and 0 <= abs_y < 3 and 0 <= abs_z < 3):
                return False
            
            if self.grid[abs_x, abs_y, abs_z] != 0:
                return False
            
            if abs_z > 0:
                has_support = (self.grid[abs_x, abs_y, abs_z - 1] != 0 or
                             any(oz < z and position[0] + ox == abs_x and position[1] + oy == abs_y
                                 for ox, oy, oz in piece_coords))
                if not has_support:
                    return False
        
        return True
    
    def _count_adjacent_blocks(self, piece_coords, position):
        adjacent_count = 0
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for x, y, z in piece_coords:
            abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
            for dx, dy, dz in directions:
                nx, ny, nz = abs_x + dx, abs_y + dy, abs_z + dz
                if (0 <= nx < 3 and 0 <= ny < 3 and 0 <= nz < 3 and 
                    self.grid[nx, ny, nz] != 0):
                    adjacent_count += 1
        
        return adjacent_count
    
    def step(self, action):
        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            return self._get_state(), self.reward_calculator.failure_penalty, True, {"timeout": True}
        
        if self.current_piece_idx >= len(self.pieces_to_place):
            return self._get_state(), self.reward_calculator.failure_penalty, True, {"error": "No more pieces"}
        
        try:
            piece_id, orient_idx, position = action
            expected_piece = self.pieces_to_place[self.current_piece_idx]
            
            if piece_id != expected_piece:
                return self._get_state(), self.reward_calculator.failure_penalty, True, {"error": "Wrong piece"}
            
            if (piece_id >= len(ALL_PIECE_ORIENTATIONS) or 
                orient_idx >= len(ALL_PIECE_ORIENTATIONS[piece_id])):
                return self._get_state(), self.reward_calculator.failure_penalty, True, {"error": "Invalid orientation"}
            
            piece_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
            
            if not self._is_valid_placement(piece_coords, position):
                return self._get_state(), self.reward_calculator.failure_penalty, True, {"error": "Invalid placement"}
            
            # 조각 배치
            for x, y, z in piece_coords:
                abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
                self.grid[abs_x, abs_y, abs_z] = piece_id + 1
            
            self.placed_pieces.append((piece_id, orient_idx, position))
            self.current_piece_idx += 1
            
            # 완료 확인
            is_success = self.current_piece_idx >= len(self.pieces_to_place)
            
            # 간단한 보상 계산
            reward = self.reward_calculator.base_reward
            if is_success:
                reward += self.reward_calculator.completion_bonus
                self.done = True
                return self._get_state(), reward, True, {"success": True}
            
            return self._get_state(), reward, False, {}
            
        except Exception as e:
            return self._get_state(), self.reward_calculator.failure_penalty, True, {"error": f"Step error: {e}"}

# ===== DQN 모델 =====
class RobustDQN(nn.Module):
    def __init__(self, state_size=42, action_size=4536):
        super().__init__()
        self.action_size = action_size
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != 42:
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.action_size, device=x.device)
            
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# ===== 행동 인덱싱 =====
PIECE_MAX_ORIENT = 24
POS_PER_GRID = 27
PIECE_SLOT = PIECE_MAX_ORIENT * POS_PER_GRID
NUM_PIECES = 7
ACTION_SIZE = NUM_PIECES * PIECE_SLOT

def encode_position(position):
    x, y, z = position
    return z * 9 + x * 3 + y

def decode_position(pos_index):
    z = pos_index // 9
    rem = pos_index % 9
    x = rem // 3
    y = rem % 3
    return (x, y, z)

def encode_action(piece_id, orient_idx, position):
    pos_idx = encode_position(position)
    return piece_id * PIECE_SLOT + orient_idx * POS_PER_GRID + pos_idx

def decode_action(action_index):
    piece_id = action_index // PIECE_SLOT
    rem = action_index % PIECE_SLOT
    orient_idx = rem // POS_PER_GRID
    pos_idx = rem % POS_PER_GRID
    return piece_id, orient_idx, decode_position(pos_idx)

# ===== 콘솔 전용 테스트 러너 =====
class SomaCubeConsoleTestRunner:
    def __init__(self):
        self.device = DEVICE
        self.node = RobotController()
        print(f"🎯 콘솔 테스트 러너 초기화: {self.device}")
        
        # 모델 초기화
        self.policy_net = RobustDQN(state_size=42, action_size=ACTION_SIZE).to(self.device)
        self.policy_net.eval()
        
        # 기본값 설정
        self.curriculum_level = 3
        self.eps = 0.0  # 테스트시에는 탐험 없이
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            print(f"📂 모델 로딩 시도: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state'])
            self.curriculum_level = checkpoint.get('curriculum_level', 1)
            print(f"✅ 모델 로드 완료! 커리큘럼 레벨: {self.curriculum_level}")
            return True
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            return False
    
    def select_action(self, state, env):
        """행동 선택 (테스트용 - 탐욕적)"""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid_actions = env.get_valid_actions()
        
        if len(valid_actions) == 0:
            return None, None, []
        
        # 행동 인덱스 변환
        valid_indices = []
        for action_tuple in valid_actions:
            try:
                idx = encode_action(*action_tuple)
                valid_indices.append(idx)
            except:
                continue
        
        if len(valid_indices) == 0:
            return None, None, []
        
        # 완전 탐욕적 선택
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)
            
            # 마스킹
            mask = torch.full((ACTION_SIZE,), float('-inf'), device=self.device)
            mask[torch.tensor(valid_indices, device=self.device)] = 0.0
            q_masked = q_values + mask
            
            chosen_index = int(torch.argmax(q_masked).item())
            chosen_action = decode_action(chosen_index)
        
        return chosen_action, chosen_index, valid_indices
    
    def somacube_test(self, level=None, num_tests=1, show_rotation_details=True):
        test_level = level or self.curriculum_level
        print(f"\n{'='*60}")
        print(f"Level {test_level} RL _ TEST SHIT!!!!!!!! ")
        print(f"{'='*60}")

        while START_FLAG == False:
            rclpy.spin_once(dsr_node, timeout_sec=0.1)
        
        success_count = 0
        
        for test in range(num_tests):
            env = SmartSomaCubeEnv(curriculum_level=test_level)
            state = env.reset()
            
            total_reward = 0
            steps = 0
            actions_taken = []
            


            while not env.done and steps < 100:
                if env.current_piece_idx >= len(env.pieces_to_place):
                    break
                
                piece_id = env.pieces_to_place[env.current_piece_idx]
                action_tuple, _, _ = self.select_action(state, env)
                
                if action_tuple is None:
                    break
                
                # 행동 실행
                next_state, reward, done, info = env.step(action_tuple)
                actions_taken.append(action_tuple)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # 성공한 경우에만 로그 출력
            if env.done and info.get("success", False):
                success_count += 1
                
                print(f"\n🎉 테스트 {test + 1}/{num_tests} 성공!")
                print("-" * 40)
                
                print(f"🎯 배치한 조각들: {[PIECE_NAMES[p] for p in env.pieces_to_place]}")
                print(f"📦 총 블록 수: {sum(len(BASE_PIECES[p]) for p in env.pieces_to_place)}")
                print()
                
                # 성공한 경우의 단계별 정보
                for i, action in enumerate(actions_taken):
                    piece_id, orient_idx, position = action
                    piece_name = PIECE_NAMES[piece_id]
                    piece_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                    
                    # 바닥층 여부 확인
                    min_z = min(position[2] + z for x, y, z in piece_coords)
                    ground_status = "바닥층" if min_z == 0 else f"{min_z}층"
                    
                    # 블록 좌표 정보
                    block_positions = []
                    for x, y, z in piece_coords:
                        abs_x, abs_y, abs_z = position[0] + x, position[1] + y, position[2] + z
                        block_positions.append(f"({abs_x},{abs_y},{abs_z})")
                    
                    print(f"🔸 단계 {i + 1}: {piece_name} 조각")
                    print(f"   📍 기준점: {position}")
                    print(f"   🔄 회전 인덱스: {orient_idx}")
                    print(f"   🏠 배치 층: {ground_status}")
                    print(f"   📦 블록 위치: {', '.join(block_positions)}")
                    
                    if show_rotation_details:
                        zyz_angles, rotation_desc = get_zyz_angles(piece_id, orient_idx)
                        print(f"   🔄 ZYZ 회전각: [{zyz_angles[0]:.1f}°, {zyz_angles[1]:.1f}°, {zyz_angles[2]:.1f}°]")
                        print(f"   🤖 로봇 명령: node.robot_control(input, {piece_id}, {zyz_angles}, {position})")

                        true_origin = get_true_visual_origin(piece_coords, position)
                        self.node.robot_control(num_tests, piece_id, zyz_angles, true_origin)
                    
                    print()
                
                print(f"📊 총 보상: {total_reward:.1f}")
                print(f"📏 총 단계: {steps}")
                
                # 로봇 제어 스크립트 요약
                if show_rotation_details and actions_taken:
                    print(f"\n🤖 로봇 제어 스크립트 요약:")
                    print("=" * 50)
                    for i, action in enumerate(actions_taken):
                        piece_id, orient_idx, position = action
                        piece_name = PIECE_NAMES[piece_id]
                        zyz_angles, rotation_desc = get_zyz_angles(piece_id, orient_idx)
                        piece_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                        min_z = min(position[2] + z for x, y, z in piece_coords)
                        level_info = f"(바닥층)" if min_z == 0 else f"({min_z}층)"
                        
                        print(f"# {i+1}. {piece_name} 조각 {level_info}")
                        print(f"node.robot_control(input, {piece_id}, {zyz_angles}, {position})")
                    print("=" * 50)
                
                if test < num_tests - 1:
                    print("\n" + "─" * 60)
        
        # 최종 통계
        print(f"\n{'='*60}")
        print(f"📊 최종 결과")
        print(f"{'='*60}")
        print(f"✅ 성공: {success_count}/{num_tests} ({success_count/num_tests:.1%})")
        print(f"❌ 실패: {num_tests - success_count}/{num_tests}")
        print(f"{'='*60}")

# ===== 메인 실행 함수 =====
def Start_callback(msg):
    global START_FLAG
    if msg.data == "1":
        START_FLAG = True

def Stop_callback(msg):
    global STOP_FLAG
    if msg.data == "1":
        STOP_FLAG = True
        sys.exit()


def main():
   print("행복해지고 싶어요 강사님")
   start_ = dsr_node.create_subscription(String, '/start_signal', Start_callback, 10)
   stop_ = dsr_node.create_subscription(String, '/stop_signal', Stop_callback, 10)
   
  
   # 테스트 러너 초기화
   runner = SomaCubeConsoleTestRunner()
    # executor를 별도 스레드에서 실행
   executor_thread = threading.Thread(target=executor.spin)
   executor_thread.daemon = True
   executor_thread.start()
  
   # 모델 로드
   if not runner.load_model(CUBE_MODEL_PATH):
       print("❌ 모델 로드 실패. 프로그램을 종료합니다.")
       return
  
   print(f"현재 모델 레벨: {runner.curriculum_level}")
   print("-" * 50)
   
   try:
       # 테스트 설정 입력
       num_tests = 100
       # show_rotation = input("회전 정보 표시? (y/n, 기본 y): ").lower() != 'n'
       # level = int(input(f"테스트 레벨 (기본 {runner.curriculum_level}): ") or str(runner.curriculum_level))
       show_rotation = True
       level = runner.curriculum_level
       
       # 상세 테스트 실행
       runner.somacube_test(level=level, num_tests=num_tests, show_rotation_details=show_rotation)
       
   except KeyboardInterrupt:
       print("\n⚠️ 테스트 중단")
   except ValueError:
       print("❌ 올바른 숫자를 입력하세요.")

# ===== 실행 =====
if __name__ == "__main__":
   main()