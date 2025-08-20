import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque, namedtuple
import time
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from od_msg.srv import SrvDepthPosition
import os
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import DR_init
from somacube.onrobot import RG
import sys

# ============================================================================
# 전역 설정
# ============================================================================
PACKAGE_NAME = "somacube"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
CUBE_MODEL_FILENAME = "best_soma_model.pth"
CUBE_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", CUBE_MODEL_FILENAME)

# Robot Configuration
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
HOME = [0, 0, 90, 0, 90, 0]
JReady = [-14.74, 6.47, 57.94, -0.03, 115.59, -14.74]

# Re-Grasp Configuration
VERTICAL_PICKUP_THRESHOLD = 1.0  # degrees
REGRASP_TEMP_HEIGHT = 100.0      # mm above current position

# Initialize ROS2 and Robot
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans, wait, DR_BASE, amovel, amovej
    from DR_common2 import posx, posj
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

# Gripper Setup
GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

# ============================================================================
# 소마큐브 조각 정의 및 회전 계산
# ============================================================================
BASE_PIECES = {
    0: np.array([[0,0,0], [1,0,0], [0,1,0]]),                    # V 조각
    1: np.array([[0,0,0], [1,0,0], [2,0,0], [2,1,0]]),           # L 조각
    2: np.array([[0,0,0], [1,0,0], [2,0,0], [1,1,0]]),           # T 조각
    3: np.array([[0,0,0], [1,0,0], [1,1,0], [2,1,0]]),           # Z 조각
    4: np.array([[0,0,0], [0,1,0], [1,1,0], [1,1,1]]),           # A 조각 (오른손)
    5: np.array([[0,0,0], [1,0,0], [1,1,0], [1,1,1]]),           # B 조각 (왼손)
    6: np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]),           # P 조각
}

def get_all_rotations():
    """모든 조각의 24가지 회전 계산"""
    all_rotations = {}
    rotation_matrices = [
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # x축 90도
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # y축 90도
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])   # z축 90도
    ]
    
    for piece_id, piece in BASE_PIECES.items():
        seen_normalized_tuples = set()
        unique_orientations_np = []
        queue = [piece]
        
        # 초기 조각 추가
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
        
        final_rotations = [p - p.min(axis=0) for p in unique_orientations_np]
        all_rotations[piece_id] = final_rotations
    
    return all_rotations

ALL_PIECE_ORIENTATIONS = get_all_rotations()

# ============================================================================
# 기하학적 유틸리티 클래스
# ============================================================================
class SquareBoardMapper:
    """3x3 격자 보드의 좌표 변환을 담당하는 클래스 - 베이스 좌표계 기준"""
    
    def __init__(self, TL, TR, BL, cube_size_mm=None):
        self.TL = np.array(TL, dtype=float)
        self.TR = np.array(TR, dtype=float)
        self.BL = np.array(BL, dtype=float)
        
        # 보드 벡터 계산
        self.u = self.TR - self.TL  # X축 방향 (가로)
        self.v = self.BL - self.TL  # Y축 방향 (세로)
        
        # 법선 벡터 계산 및 검증 (Z축 방향)
        n = np.cross(self.u, self.v)
        if np.linalg.norm(n) < 1e-10:
            raise ValueError("TL, TR, BL are collinear - cannot form valid plane")
        
        n /= np.linalg.norm(n)
        self.n_up = n if n[2] > 0 else -n  # Z축이 위쪽이 되도록
        
        # 격자 단위 벡터 (3x3 격자)
        self.du = self.u / 3.0  # X축 단위 벡터
        self.dv = self.v / 3.0  # Y축 단위 벡터
        
        # 큐브 크기 결정
        side_length = 0.5 * (np.linalg.norm(self.u) + np.linalg.norm(self.v))
        self.s = float(cube_size_mm) if cube_size_mm is not None else side_length / 3.0
        
        # 베이스 좌표계 구성 (정규화된 축)
        self.base_x = self.u / np.linalg.norm(self.u)  # 정규화된 X축
        self.base_y = self.v / np.linalg.norm(self.v)  # 정규화된 Y축  
        self.base_z = self.n_up                        # 정규화된 Z축
        
        # 베이스 좌표계 변환 행렬 구성
        self.base_transform = np.eye(4)
        self.base_transform[:3, 0] = self.base_x
        self.base_transform[:3, 1] = self.base_y
        self.base_transform[:3, 2] = self.base_z
        self.base_transform[:3, 3] = self.TL
        
        # 검증
        u_len = np.linalg.norm(self.u)
        v_len = np.linalg.norm(self.v)
        if abs(u_len - v_len) > 5.0:  # 5mm 이상 차이
            print(f"WARNING: Board is not square! u_len={u_len:.2f}, v_len={v_len:.2f}")
        
        print(f"[Board] Base coordinate system established:")
        print(f"  X-axis: {np.round(self.base_x, 3)}")
        print(f"  Y-axis: {np.round(self.base_y, 3)}")
        print(f"  Z-axis: {np.round(self.base_z, 3)}")

    def grid_to_base_coords(self, i, j, k):
        """격자 좌표를 베이스 좌표계 기준으로 변환"""
        # 격자에서의 로컬 좌표
        local_x = (i + 0.5) * (np.linalg.norm(self.u) / 3.0)
        local_y = (j + 0.5) * (np.linalg.norm(self.v) / 3.0)
        local_z = (k + 0.5) * self.s
        
        # 베이스 좌표계 기준 위치
        base_pos = (self.TL + 
                   local_x * self.base_x + 
                   local_y * self.base_y + 
                   local_z * self.base_z)
        
        return base_pos.tolist()

    def get_base_orientation(self):
        """베이스 좌표계의 오일러각 반환 (ZYZ)"""
        # 베이스 변환 행렬에서 회전 부분 추출
        rotation_matrix = self.base_transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        return rotation.as_euler('zyz', degrees=True).tolist()

    def voxel_to_base_pose(self, x, y, z, orientation_euler_zyz=None):
        """복셀 좌표를 베이스 좌표계 기준 6DOF 포즈로 변환"""
        # 위치 계산
        position = self.grid_to_base_coords(x, y, z)
        
        # 오리엔테이션 계산 (베이스 좌표계 기준)
        if orientation_euler_zyz is None:
            orientation = self.get_base_orientation()
        else:
            # 베이스 오리엔테이션에 추가 회전 적용
            base_rotation = R.from_matrix(self.base_transform[:3, :3])
            additional_rotation = R.from_euler('zyz', orientation_euler_zyz, degrees=True)
            combined_rotation = base_rotation * additional_rotation
            orientation = combined_rotation.as_euler('zyz', degrees=True).tolist()
        
        return position + orientation

    def cell_center_world(self, i, j, k):
        """격자 셀 중심의 월드 좌표 반환 (하위 호환성)"""
        return self.grid_to_base_coords(i, j, k)

    def voxel_world(self, x, y, z):
        """복셀 좌표를 월드 좌표로 변환 (하위 호환성)"""
        return self.grid_to_base_coords(x, y, z)

# ============================================================================
# Re-Grasp 로직 및 유틸리티
# ============================================================================
class ReGraspPlanner:
    """Re-Grasp 계획을 담당하는 클래스"""
    
    @staticmethod
    def release_reference_base(board: SquareBoardMapper, piece_voxels_np, pos_xyz):
        """릴리즈 기준점 계산 (베이스 좌표계 기준, 지지 복셀들의 평균)"""
        voxels = np.asarray(piece_voxels_np)
        zmin = voxels[:, 2].min()
        bottom_voxels = voxels[voxels[:, 2] == zmin]
        
        # 베이스 좌표계 기준으로 중심점들 계산
        centers = [board.grid_to_base_coords(pos_xyz[0] + x, pos_xyz[1] + y, pos_xyz[2] + z) 
                  for x, y, z in bottom_voxels]
        return np.mean(np.array(centers), axis=0).tolist()

    @staticmethod
    def calculate_base_aligned_rotation(board: SquareBoardMapper, base_coords, target_coords):
        """베이스 좌표계에 정렬된 회전 계산"""
        try:
            # 기본 회전 계산
            rotation = calculate_rotation(base_coords, target_coords)
            if rotation is None:
                return [0.0, 0.0, 0.0]
            
            # 베이스 좌표계에 맞춰 회전 조정
            raw_euler = rotation.as_euler('zyz', degrees=True)
            
            # 베이스 좌표계의 방향을 고려한 스냅
            base_orientation = board.get_base_orientation()
            
            # 베이스 기준으로 회전 보정
            base_rot = R.from_euler('zyz', base_orientation, degrees=True)
            piece_rot = R.from_euler('zyz', raw_euler, degrees=True)
            
            # 베이스 좌표계 기준으로 정렬된 회전
            aligned_rot = base_rot.inv() * piece_rot * base_rot
            aligned_euler = aligned_rot.as_euler('zyz', degrees=True)
            
            return ReGraspPlanner.snap90_zyz(aligned_euler)
            
        except Exception as e:
            print(f"Error in base-aligned rotation calculation: {e}")
            return [0.0, 0.0, 0.0]

    @staticmethod
    def decide_execution_plan_base_aligned(board: SquareBoardMapper, base_coords, target_coords, 
                                          piece_orientations, threshold_deg=VERTICAL_PICKUP_THRESHOLD):
        """베이스 좌표계 기준 실행 계획 결정"""
        # 베이스 정렬된 회전 계산
        aligned_euler = ReGraspPlanner.calculate_base_aligned_rotation(board, base_coords, target_coords)
        
        print(f"     [Base-Aligned] Raw rotation: {aligned_euler}")
        
        # 수직 픽업 가능성 판정 (베이스 Z축 기준)
        if ReGraspPlanner.is_vertical_pickup(aligned_euler, threshold_deg):
            return {
                "type": "direct",
                "rotation": aligned_euler,
                "reason": "vertical_pickup_ok_base_aligned"
            }
        
        # Re-Grasp 필요: 안정한 중간자세 찾기
        stable_indices = ReGraspPlanner.stable_orientation_indices(piece_orientations)
        if stable_indices:
            # 베이스 좌표계 기준 수직 픽업
            base_vertical = [0.0, 0.0, 0.0]  # 베이스 Z축과 평행
            
            return {
                "type": "regrasp",
                "leg1_rotation": base_vertical,
                "leg2_rotation": aligned_euler,
                "intermediate_idx": stable_indices[0],
                "reason": "vertical_constraint_violation_base_aligned"
            }
        
        return {
            "type": "fail",
            "reason": "no_stable_intermediate_pose"
        }

    @staticmethod
    def snap90_zyz(euler_zyz_deg):
        """ZYZ 오일러각을 90도 단위로 스냅"""
        z1, y, z2 = euler_zyz_deg
        z1_snap = 90.0 * round(z1 / 90.0)
        y_snap = 90.0 * round(y / 90.0)
        z2_snap = 90.0 * round(z2 / 90.0)
        
        # Y축 범위 제한 (ZYZ에서 Y는 [0, 180] 범위)
        y_snap = max(0, min(180, y_snap))
        
        return [z1_snap, y_snap, z2_snap]

    @staticmethod
    def is_vertical_pickup(euler_zyz_deg, threshold_deg=VERTICAL_PICKUP_THRESHOLD):
        """수직 픽업 가능 여부 판정 (Y축 각도 기준)"""
        return abs(euler_zyz_deg[1]) <= threshold_deg

    @staticmethod
    def stable_orientation_indices(piece_orientations):
        """안정한 중간자세 인덱스들 반환"""
        stable_indices = []
        
        for idx, voxels in enumerate(piece_orientations):
            voxels = np.asarray(voxels)
            if len(voxels) == 0:
                continue
                
            zmin = voxels[:, 2].min()
            bottom_points = voxels[voxels[:, 2] == zmin][:, :2]  # XY만
            cog_xy = voxels.mean(axis=0)[:2]  # 질량중심의 XY 투영
            
            # 간단한 안정성 검사: CoG가 지지 영역 내부에 있는지
            if len(bottom_points) >= 1:
                x_min, x_max = bottom_points[:, 0].min(), bottom_points[:, 0].max()
                y_min, y_max = bottom_points[:, 1].min(), bottom_points[:, 1].max()
                
                if (x_min <= cog_xy[0] <= x_max and y_min <= cog_xy[1] <= y_max):
                    stable_indices.append(idx)
        
        return stable_indices

    @staticmethod
    def decide_execution_plan(euler_zyz_deg, piece_orientations, threshold_deg=VERTICAL_PICKUP_THRESHOLD):
        """실행 계획 결정: 직접 실행 vs Re-Grasp"""
        snapped_euler = ReGraspPlanner.snap90_zyz(euler_zyz_deg)
        
        # 수직 픽업 가능하면 직접 실행
        if ReGraspPlanner.is_vertical_pickup(snapped_euler, threshold_deg):
            return {
                "type": "direct",
                "rotation": snapped_euler,
                "reason": "vertical_pickup_ok"
            }
        
        # Re-Grasp 필요: 안정한 중간자세 찾기
        stable_indices = ReGraspPlanner.stable_orientation_indices(piece_orientations)
        if stable_indices:
            return {
                "type": "regrasp",
                "leg1_rotation": [0.0, 0.0, 0.0],  # 수직 픽업
                "leg2_rotation": snapped_euler,     # 목표 회전
                "intermediate_idx": stable_indices[0],
                "reason": "vertical_constraint_violation"
            }
        
        return {
            "type": "fail",
            "reason": "no_stable_intermediate_pose"
        }

# ============================================================================
# 회전 계산 유틸리티
# ============================================================================
def calculate_rotation(base_coords, target_coords):
    """base_coords를 target_coords로 변환하는 회전 계산"""
    try:
        base_points = np.asarray(base_coords, dtype=float)
        target_points = np.asarray(target_coords, dtype=float)
        
        if base_points.shape != target_points.shape or base_points.shape[1] != 3:
            return None
        
        # 중심점 제거 (병진 성분 제거)
        base_centered = base_points - base_points.mean(axis=0)
        target_centered = target_points - target_points.mean(axis=0)
        
        # 회전 계산 (base -> target)
        rotation, rmsd = R.align_vectors(target_centered, base_centered)
        
        # 유효성 검사
        rotvec = rotation.as_rotvec()
        if not np.all(np.isfinite(rotvec)) or np.linalg.norm(rotvec) < 1e-6:
            return None
            
        return rotation
        
    except Exception as e:
        print(f"Error in calculate_rotation: {e}")
        return None

# ============================================================================
# 유틸리티 함수들
# ============================================================================
def up_pos(position, axis, value):
    """위치의 특정 축에 값을 더함"""
    pos = position.copy()
    pos[axis] += value
    return posx(pos)

def apply_rotation_manually(start_pose, zyz_delta):
    """수동으로 ZYZ 회전 적용"""
    start_euler = start_pose[3:]
    r_start = R.from_euler('zyz', start_euler, degrees=True)
    r_delta = R.from_euler('zyz', zyz_delta, degrees=True)
    r_final = r_start * r_delta
    final_euler = r_final.as_euler('zyz', degrees=True)
    return start_pose[:3] + final_euler.tolist()

# ============================================================================
# 로봇 컨트롤러 클래스
# ============================================================================
class RobotController(Node):
    """로봇 제어를 담당하는 메인 클래스"""
    
    def __init__(self):
        super().__init__("somacube")
        self.init_robot()
        self.setup_depth_client()
        self.regrasp_planner = ReGraspPlanner()

    def setup_depth_client(self):
        """깊이 센서 클라이언트 설정"""
        self.depth_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        while not self.depth_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for depth position service...")
        self.depth_request = SrvDepthPosition.Request()

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        """로봇 포즈를 변환 행렬로 변환"""
        rotation_matrix = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = [x, y, z]
        return transform_matrix

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """카메라 좌표를 로봇 베이스 좌표로 변환"""
        try:
            gripper2cam = np.load(gripper2cam_path)
            coord = np.append(np.array(camera_coords), 1)  # 동차좌표
            
            x, y, z, rx, ry, rz = robot_pos
            base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
            
            base2cam = base2gripper @ gripper2cam
            transformed_coord = np.dot(base2cam, coord)
            
            return transformed_coord[:3]
        except Exception as e:
            self.get_logger().error(f"Transform error: {e}")
            return None

    def init_robot(self):
        """로봇 초기 자세로 이동"""
        try:
            movej(JReady, vel=VELOCITY, acc=ACC)
            gripper.open_gripper()
            mwait()
        except Exception as e:
            self.get_logger().error(f"Robot initialization error: {e}")

    def execute_regrasp_sequence(self, target_pos, plan):
        """Re-Grasp 시퀀스 실행"""
        self.get_logger().info("Executing re-grasp sequence...")
        
        try:
            # 1단계: 수직으로 픽업
            self.get_logger().info("Re-grasp Step 1: Vertical pickup")
            temp_pos = target_pos.copy()
            movel(temp_pos, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 중간 높이로 이동
            temp_pos_up = up_pos(temp_pos, 2, REGRASP_TEMP_HEIGHT)
            movel(temp_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 임시 놓기 위치로 이동 (여기서는 같은 위치에 회전만 적용)
            # 실제로는 안전한 임시 위치를 정의해야 함
            temp_release_pos = apply_rotation_manually(temp_pos_up, plan["leg1_rotation"])
            movel(temp_release_pos, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 2단계: 목표 회전으로 다시 픽업
            self.get_logger().info("Re-grasp Step 2: Target rotation pickup")
            final_pos_up = up_pos(temp_pos, 2, REGRASP_TEMP_HEIGHT)
            final_pos_rotated = apply_rotation_manually(final_pos_up, plan["leg2_rotation"])
            
            # 충돌 방지 루틴
            if 70 <= final_pos_rotated[4] <= 90 or -90 <= final_pos_rotated[4] <= -70:
                correction_delta = [0, 0, 180.0]
                final_pos_rotated = apply_rotation_manually(final_pos_rotated, correction_delta)
            
            movel(final_pos_rotated, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 최종 위치로 이동
            final_pos = apply_rotation_manually(temp_pos, plan["leg2_rotation"])
            movel(final_pos, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f"Re-grasp execution error: {e}")

    def execute_direct_sequence(self, target_pos, rotation):
        """직접 실행 시퀀스"""
        self.get_logger().info("Executing direct sequence...")
        
        try:
            # 픽업
            temp_pos = target_pos.copy()
            temp_pos[3] -= 10  # 원래 코드의 오프셋 유지
            
            movel(temp_pos, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 상승 후 회전
            target_pos_up = up_pos(temp_pos, 2, 300)
            target_pos_up = up_pos(target_pos_up, 0, 100)
            movel(target_pos_up, vel=VELOCITY, acc=ACC)
            
            target_pos_rotation = apply_rotation_manually(target_pos_up, rotation)
            
            # 충돌 방지 루틴
            if 70 <= target_pos_rotation[4] <= 90 or -90 <= target_pos_rotation[4] <= -70:
                correction_delta = [0, 0, 180.0]
                target_pos_rotation = apply_rotation_manually(target_pos_rotation, correction_delta)
            
            movel(target_pos_rotation, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 릴리즈
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f"Direct execution error: {e}")

    def execute_base_aligned_sequence(self, board, target_grid_pos, execution_plan):
        """베이스 좌표계 정렬된 시퀀스 실행"""
        try:
            # 격자 좌표를 베이스 좌표계 6DOF 포즈로 변환
            i, j, k = target_grid_pos
            
            if execution_plan["type"] == "direct":
                target_pose_6d = board.voxel_to_base_pose(i, j, k, execution_plan["rotation"])
                self.get_logger().info(f"Direct execution to base pose: {np.round(target_pose_6d, 2).tolist()}")
                self.execute_direct_sequence(target_pose_6d, execution_plan["rotation"])
                
            elif execution_plan["type"] == "regrasp":
                # 1단계: 베이스 정렬된 수직 픽업
                pickup_pose_6d = board.voxel_to_base_pose(i, j, k, execution_plan["leg1_rotation"])
                
                # 2단계: 베이스 정렬된 목표 회전
                final_pose_6d = board.voxel_to_base_pose(i, j, k, execution_plan["leg2_rotation"])
                
                self.get_logger().info(f"Re-grasp execution:")
                self.get_logger().info(f"  Pickup pose: {np.round(pickup_pose_6d, 2).tolist()}")
                self.get_logger().info(f"  Final pose: {np.round(final_pose_6d, 2).tolist()}")
                
                # Re-grasp 시퀀스 실행 (수정된 파라미터)
                regrasp_plan = {
                    "leg1_rotation": execution_plan["leg1_rotation"],
                    "leg2_rotation": execution_plan["leg2_rotation"],
                    "pickup_pose": pickup_pose_6d,
                    "final_pose": final_pose_6d
                }
                self.execute_regrasp_sequence_base_aligned(regrasp_plan)
                
        except Exception as e:
            self.get_logger().error(f"Base-aligned execution error: {e}")

    def execute_regrasp_sequence_base_aligned(self, plan):
        """베이스 정렬된 Re-Grasp 시퀀스 실행"""
        self.get_logger().info("Executing base-aligned re-grasp sequence...")
        
        try:
            # 1단계: 베이스 정렬된 수직 픽업
            pickup_pose = plan["pickup_pose"]
            
            self.get_logger().info("Re-grasp Step 1: Base-aligned vertical pickup")
            movel(pickup_pose, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 중간 높이로 이동 (베이스 Z축 방향)
            intermediate_pose = pickup_pose.copy()
            intermediate_pose[2] += REGRASP_TEMP_HEIGHT  # 베이스 Z축 방향으로 상승
            movel(intermediate_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 임시 놓기 (베이스 좌표계 기준)
            temp_release_pose = apply_rotation_manually(intermediate_pose, plan["leg1_rotation"])
            movel(temp_release_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 2단계: 베이스 정렬된 목표 회전으로 재픽업
            self.get_logger().info("Re-grasp Step 2: Base-aligned target rotation")
            
            final_pose = plan["final_pose"]
            
            # 상승된 위치에서 목표 회전 적용
            elevated_final_pose = final_pose.copy()
            elevated_final_pose[2] += REGRASP_TEMP_HEIGHT
            
            movel(elevated_final_pose, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 최종 배치 (베이스 좌표계 기준)
            movel(final_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f"Base-aligned re-grasp execution error: {e}")

    def robot_control_base_aligned(self, input_data, piece_id, target_grid_pos, execution_plan, board):
        """베이스 좌표계 기준 로봇 제어 - YOLO 위치와 격자 위치 결합"""
        if input_data.lower() == "q":
            self.get_logger().info("Quit the program...")
            sys.exit()

        if not input_data:
            return

        try:
            # 1. YOLO로 실제 블록 위치 감지
            self.depth_request.target = str(piece_id)
            self.get_logger().info("Calling depth position service with YOLO")
            depth_future = self.depth_client.call_async(self.depth_request)
            rclpy.spin_until_future_complete(self, depth_future)

            if not depth_future.result():
                self.get_logger().error("Failed to get depth position")
                return

            yolo_result = depth_future.result().depth_position.tolist()
            self.get_logger().info(f"YOLO detected position: {yolo_result}")
            
            if sum(yolo_result) == 0:
                self.get_logger().warning("No target position detected by YOLO")
                return

            # 2. YOLO 결과를 베이스 좌표계로 변환
            gripper2cam_path = os.path.join(PACKAGE_PATH, "resource", "T_gripper2camera.npy")
            robot_posx = get_current_posx()[0]
            yolo_base_coord = self.transform_to_base(yolo_result, gripper2cam_path, robot_posx)

            if yolo_base_coord is None:
                self.get_logger().error("YOLO coordinate transformation failed")
                return

            # 깊이 보정
            if yolo_base_coord[2] and sum(yolo_base_coord) != 0:
                yolo_base_coord[2] += -5  # DEPTH_OFFSET
                yolo_base_coord[2] = max(yolo_base_coord[2], 2)  # MIN_DEPTH

            # 3. 목표 격자 위치의 베이스 좌표 계산 (릴리즈 위치)
            i, j, k = target_grid_pos
            target_base_position = board.grid_to_base_coords(i, j, k)
            
            self.get_logger().info(f"YOLO pickup position: {np.round(yolo_base_coord, 2).tolist()}")
            self.get_logger().info(f"Target release position: {np.round(target_base_position, 2).tolist()}")

            # 4. 실행 계획에 따라 분기
            if execution_plan["type"] == "direct":
                self.execute_direct_sequence_hybrid(
                    yolo_base_coord, target_base_position, execution_plan["rotation"]
                )
            elif execution_plan["type"] == "regrasp":
                self.execute_regrasp_sequence_hybrid(
                    yolo_base_coord, target_base_position, execution_plan
                )
            else:
                self.get_logger().error(f"Execution failed: {execution_plan['reason']}")

        except Exception as e:
            self.get_logger().error(f"Base-aligned robot control error: {e}")
        finally:
            self.init_robot()

    def execute_direct_sequence_hybrid(self, pickup_pos, release_pos, rotation):
        """YOLO 픽업 + 베이스 릴리즈 직접 실행"""
        self.get_logger().info("Executing hybrid direct sequence...")
        
        try:
            # 1. YOLO 위치에서 픽업
            pickup_pose = list(pickup_pos) + [0, 0, 0]  # 현재 로봇 오리엔테이션 유지
            pickup_pose[3] -= 10  # 피치 오프셋 (기존 코드 유지)
            
            self.get_logger().info(f"Picking up at: {np.round(pickup_pose[:3], 2)}")
            movel(pickup_pose, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 2. 상승
            pickup_pose_up = up_pos(pickup_pose, 2, 300)  # 300mm 상승
            pickup_pose_up = up_pos(pickup_pose_up, 0, 100)  # 100mm 전진
            movel(pickup_pose_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 3. 목표 위치로 이동하면서 회전 적용
            release_pose = list(release_pos) + rotation
            release_pose_up = list(release_pos) + rotation
            release_pose_up[2] += 300  # 목표 위치 위 300mm
            
            # 충돌 방지 루틴
            if 70 <= release_pose_up[4] <= 90 or -90 <= release_pose_up[4] <= -70:
                correction_delta = [0, 0, 180.0]
                release_pose_up = apply_rotation_manually(release_pose_up, correction_delta)
                release_pose = apply_rotation_manually(release_pose, correction_delta)
            
            self.get_logger().info(f"Moving to release position: {np.round(release_pose_up[:3], 2)}")
            movel(release_pose_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 4. 하강하여 배치
            self.get_logger().info(f"Placing at: {np.round(release_pose[:3], 2)}")
            movel(release_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 5. 릴리즈
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f"Hybrid direct execution error: {e}")

    def execute_regrasp_sequence_hybrid(self, pickup_pos, release_pos, plan):
        """YOLO 픽업 + 베이스 릴리즈 Re-Grasp 실행"""
        self.get_logger().info("Executing hybrid re-grasp sequence...")
        
        try:
            # 1단계: YOLO 위치에서 수직 픽업
            pickup_pose = list(pickup_pos) + plan["leg1_rotation"]
            
            self.get_logger().info("Re-grasp Step 1: Pickup from YOLO position")
            movel(pickup_pose, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 중간 높이로 이동
            intermediate_pose = pickup_pose.copy()
            intermediate_pose[2] += REGRASP_TEMP_HEIGHT
            movel(intermediate_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 임시 안전 위치에서 놓기 (픽업 위치 근처)
            temp_release_pose = apply_rotation_manually(intermediate_pose, [0, 0, 0])
            movel(temp_release_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 2단계: 목표 회전으로 재픽업
            self.get_logger().info("Re-grasp Step 2: Re-pickup with target rotation")
            
            # 같은 위치에서 회전된 자세로 다시 픽업
            repickup_pose = list(pickup_pos) + plan["leg2_rotation"]
            repickup_pose[2] += REGRASP_TEMP_HEIGHT
            
            movel(repickup_pose, vel=VELOCITY, acc=ACC)
            mwait()
            gripper.close_gripper()
            
            while gripper.get_status()[0]:
                time.sleep(0.5)
            
            # 3단계: 목표 위치로 이동하여 배치
            final_pose = list(release_pos) + plan["leg2_rotation"]
            final_pose_up = final_pose.copy()
            final_pose_up[2] += 200  # 목표 위치 위 200mm
            
            self.get_logger().info("Re-grasp Step 3: Move to final position")
            movel(final_pose_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            # 최종 배치
            movel(final_pose, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            while gripper.get_status()[0]:
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f"Hybrid re-grasp execution error: {e}")

# ============================================================================
# 강화학습 관련 클래스들 (기존 유지)
# ============================================================================
class ActionMapper:
    def __init__(self):
        self.action_to_index = {}
        self.index_to_action = {}
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

    def action_to_idx(self, action):
        return self.action_to_index.get(action, -1)

    def idx_to_action(self, idx):
        return self.index_to_action.get(idx, None)

class SomaCubeEnv:
    def __init__(self, action_mapper):
        self.grid_shape = (3, 3, 3)
        self.action_mapper = action_mapper
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_shape, dtype=int)
        self.pieces_to_place = random.sample(list(range(7)), 7)
        self.current_piece_idx = self.pieces_to_place.pop(0)
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = np.zeros(27 + 7)
        state[:27] = self.grid.flatten()
        state[27 + self.current_piece_idx] = 1
        return state

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
            if abs_z > 0 and self.grid[position[0] + x, position[1] + y, abs_z - 1] != 0:
                return True
        return False

    def get_possible_actions(self):
        possible_action_indices = []
        orientations = ALL_PIECE_ORIENTATIONS[self.current_piece_idx]
        
        for orient_idx, piece_coords in enumerate(orientations):
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        if (self._is_valid_placement(piece_coords, (x, y, z)) and 
                            self._is_supported(piece_coords, (x, y, z))):
                            action = (self.current_piece_idx, orient_idx, (x, y, z))
                            action_idx = self.action_mapper.action_to_idx(action)
                            if action_idx != -1:
                                possible_action_indices.append(action_idx)
        return possible_action_indices

    def step(self, action_idx):
        action = self.action_mapper.idx_to_action(action_idx)
        if action is None or action[0] != self.current_piece_idx:
            return self._get_state(), -5.0, True, {"error": "Invalid action"}

        piece_id, orient_idx, position = action
        piece_coords = ALL_PIECE_ORIENTATIONS[self.current_piece_idx][orient_idx]
        
        if (not self._is_valid_placement(piece_coords, position) or 
            not self._is_supported(piece_coords, position)):
            return self._get_state(), -5.0, True, {"error": "Invalid placement"}

        for x, y, z in piece_coords:
            self.grid[position[0] + x, position[1] + y, position[2] + z] = self.current_piece_idx + 1

        if not self.pieces_to_place:
            return self._get_state(), 100.0, True, {"success": True}

        self.current_piece_idx = self.pieces_to_place.pop(0)
        if not self.get_possible_actions():
            return self._get_state(), -10.0, True, {"error": "Stuck"}

        return self._get_state(), 1.0, False, {}

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.network(x.float())

# ============================================================================
# 메인 실행 함수
# ============================================================================
def main():
    """메인 실행 함수"""
    
    try:
        # 시스템 초기화
        node = RobotController()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_mapper = ActionMapper()
        env = SomaCubeEnv(action_mapper)
        state_size, action_size = 34, action_mapper.total_actions
        policy_net = DQN(state_size, action_size).to(device)
        
        # 모델 로딩
        print(f"🤖 Loading trained model from '{CUBE_MODEL_PATH}'")
        try:
            policy_net.load_state_dict(torch.load(CUBE_MODEL_PATH, map_location=device))
            policy_net.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at '{CUBE_MODEL_PATH}'. Please run training script first.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # 보드 매퍼 초기화
        try:
            board = SquareBoardMapper(
                TL=(424.850, 78.830, 12.4),
                TR=(499.900, 78.830, 12.4),
                BL=(424.850, 3.830, 12.4),
                cube_size_mm=25.0
            )
            print(f"[Board] Initialized - s={board.s:.2f}mm, "
                  f"|du|={np.linalg.norm(board.du):.2f}, "
                  f"|dv|={np.linalg.norm(board.dv):.2f}")
        except ValueError as e:
            print(f"Board initialization failed: {e}")
            return

        print("Starting test...")
        print("-" * 80)

        # 메인 테스트 루프
        while True:
            user_input = input("Enter 1 to start assembly, 2 for manual mode, q to quit: ")
            
            if user_input.lower() == 'q':
                break
                
            if user_input not in ['1', '2']:
                continue

            if user_input == '2':
                # 수동 모드: 개별 조각 테스트
                piece_id = int(input("Enter piece ID (0-6): "))
                if piece_id not in range(7):
                    print("Invalid piece ID")
                    continue
                    
                orient_idx = int(input(f"Enter orientation index (0-{len(ALL_PIECE_ORIENTATIONS[piece_id])-1}): "))
                if orient_idx not in range(len(ALL_PIECE_ORIENTATIONS[piece_id])):
                    print("Invalid orientation index")
                    continue
                    
                x = int(input("Enter X position (0-2): "))
                y = int(input("Enter Y position (0-2): "))
                z = int(input("Enter Z position (0-2): "))
                
                if not all(0 <= val <= 2 for val in [x, y, z]):
                    print("Invalid position")
                    continue
                
                # 수동 실행
                manual_step = (piece_id, orient_idx, (x, y, z))
                print(f"🔧 Manual execution: {manual_step}")
                
                # 수동 모드 실행 로직
                base_coords = BASE_PIECES[piece_id]
                target_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                rotation = calculate_rotation(base_coords, target_coords)
                
                if rotation is None:
                    raw_euler_zyz = [0.0, 0.0, 0.0]
                else:
                    raw_euler_zyz = rotation.as_euler('zyz', degrees=True)
                
                piece_orientations = ALL_PIECE_ORIENTATIONS[piece_id]
                execution_plan = ReGraspPlanner.decide_execution_plan_base_aligned(
                    board, base_coords, target_coords, piece_orientations, VERTICAL_PICKUP_THRESHOLD
                )
                
                print(f"     Execution plan: {execution_plan['type']}")
                
                try:
                    node.robot_control_base_aligned("1", piece_id, (x, y, z), execution_plan, board)
                    print(f"     ✅ Manual execution completed")
                except Exception as e:
                    print(f"     ❌ Manual execution failed: {e}")
                
                continue

            # 자동 모드 - 성공할 때까지 무한 루프 (안전장치 추가)
            print("🎯 Starting assembly planning - will retry until success...")
            
            attempt = 0
            max_attempts = 50  # 무한 루프 방지를 위한 상한
            
            while attempt < max_attempts:  # 안전장치 추가
                attempt += 1
                print(f"\n🔄 Planning attempt {attempt}")
                
                try:
                    # 환경 리셋
                    state = env.reset()
                    done = False
                    step_count = 0
                    total_reward = 0
                    solution_path = []
                    
                    # 환경 상태 검증
                    if state is None or len(state) != state_size:
                        print(f"   ❌ Invalid environment state: {state}")
                        continue
                    
                    print(f"   Pieces order: {env.pieces_to_place + [env.current_piece_idx]}")

                    # DQN 기반 조립 시퀀스 생성
                    while not done and step_count < 100:
                        possible_actions = env.get_possible_actions()
                        
                        if not possible_actions:
                            print(f"   ❌ No possible actions at step {step_count + 1}")
                            break

                        # DQN으로 최적 행동 선택
                        try:
                            with torch.no_grad():
                                state_tensor = torch.tensor([state], device=device, dtype=torch.float)
                                q_values = policy_net(state_tensor)[0]
                                best_action_idx = max(possible_actions, 
                                                    key=lambda idx: q_values[idx].item())

                            action = action_mapper.idx_to_action(best_action_idx)
                            if action is None:
                                print(f"   ❌ Invalid action index: {best_action_idx}")
                                break
                                
                            solution_path.append(action)
                            state, reward, done, info = env.step(best_action_idx)
                            total_reward += reward
                            step_count += 1
                            
                        except Exception as e:
                            print(f"   ❌ DQN step error: {e}")
                            break

                    # 성공 여부 확인
                    if done and "success" in info:
                        print(f"✅ Assembly Planning: SUCCESS on attempt {attempt}!")
                        print(f"   Final reward: {total_reward:.1f}, Steps: {step_count}")
                        break  # 성공하면 루프 탈출
                    else:
                        print(f"   ❌ Failed - Reward: {total_reward:.1f}, Steps: {step_count}, Reason: {info.get('error', 'Unknown')}")
                        
                except Exception as e:
                    print(f"   ❌ Planning attempt {attempt} crashed: {e}")
                    continue

            # 최대 시도 횟수 도달시 처리
            if attempt >= max_attempts:
                print(f"⚠️ Reached maximum attempts ({max_attempts}). Using fallback strategy...")
                # 간단한 폴백: 첫 번째 조각만 배치
                try:
                    env.reset()
                    solution_path = [(0, 0, (0, 0, 0))]  # 조각 0을 (0,0,0)에 기본 방향으로
                    print(f"✅ Fallback: Single piece placement")
                except Exception as e:
                    print(f"❌ Fallback also failed: {e}")
                    return

            # 성공한 solution_path로 로봇 실행
            if not solution_path:
                print("❌ No valid solution path generated")
                return
                
            print(f"\n🔧 Executing assembly sequence ({len(solution_path)} steps):")
            
            for i, step in enumerate(solution_path):
                if step is None:
                    print(f"   - Step {i+1}: Invalid action occurred. Skipping.")
                    continue

                try:
                    piece_id, orient_idx, pos = step
                    
                    # 입력 검증
                    if not (0 <= piece_id < 7):
                        print(f"   - Step {i+1}: Invalid piece_id {piece_id}. Skipping.")
                        continue
                        
                    if not (0 <= orient_idx < len(ALL_PIECE_ORIENTATIONS[piece_id])):
                        print(f"   - Step {i+1}: Invalid orient_idx {orient_idx}. Skipping.")
                        continue
                        
                    if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
                        print(f"   - Step {i+1}: Invalid position {pos}. Skipping.")
                        continue
                    
                    print(f"\n   - Step {i+1}: Piece {piece_id} at position {pos}")

                    # 회전 계산
                    base_coords = BASE_PIECES[piece_id]
                    target_coords = ALL_PIECE_ORIENTATIONS[piece_id][orient_idx]
                    rotation = calculate_rotation(base_coords, target_coords)

                    # 로깅 (축-각 정보)
                    if rotation is None:
                        print("     🔄 Rotation: No rotation needed")
                        raw_euler_zyz = [0.0, 0.0, 0.0]
                    else:
                        try:
                            rotvec = rotation.as_rotvec()
                            angle_deg = np.rad2deg(np.linalg.norm(rotvec))
                            if angle_deg > 0:
                                axis = rotvec / np.linalg.norm(rotvec)
                                print(f"     🔄 Rotation: Axis {np.round(axis, 2)} by {angle_deg:.1f}°")
                            raw_euler_zyz = rotation.as_euler('zyz', degrees=True)
                        except Exception as e:
                            print(f"     ❌ Rotation calculation error: {e}")
                            raw_euler_zyz = [0.0, 0.0, 0.0]

                    # Re-Grasp 계획 수립 (베이스 좌표계 기준)
                    piece_orientations = ALL_PIECE_ORIENTATIONS[piece_id]
                    execution_plan = ReGraspPlanner.decide_execution_plan_base_aligned(
                        board, base_coords, target_coords, piece_orientations, VERTICAL_PICKUP_THRESHOLD
                    )

                    # 실행 계획 검증
                    if not isinstance(execution_plan, dict) or "type" not in execution_plan:
                        print(f"     ❌ Invalid execution plan: {execution_plan}")
                        continue

                    # 릴리즈 기준점 계산 (베이스 좌표계 기준)
                    try:
                        release_ref = ReGraspPlanner.release_reference_base(board, target_coords, pos)
                        print(f"     📍 Release reference (base): {np.round(release_ref, 2).tolist()}")
                    except Exception as e:
                        print(f"     ❌ Release reference calculation error: {e}")

                    # 실행 계획 로깅
                    if execution_plan["type"] == "direct":
                        print(f"     ✅ Plan: DIRECT execution (base-aligned)")
                        print(f"        ZYZ rotation: {execution_plan.get('rotation', [0,0,0])}")
                        print(f"        Reason: {execution_plan.get('reason', 'unknown')}")
                    elif execution_plan["type"] == "regrasp":
                        print(f"     🔄 Plan: RE-GRASP execution (base-aligned)")
                        print(f"        Leg1 (pickup): {execution_plan.get('leg1_rotation', [0,0,0])}")
                        print(f"        Leg2 (final): {execution_plan.get('leg2_rotation', [0,0,0])}")
                        print(f"        Intermediate pose index: {execution_plan.get('intermediate_idx', 0)}")
                        print(f"        Reason: {execution_plan.get('reason', 'unknown')}")
                    else:
                        print(f"     ❌ Plan: EXECUTION FAILED")
                        print(f"        Reason: {execution_plan.get('reason', 'unknown')}")
                        print("        Skipping this piece...")
                        continue

                    # 베이스 좌표계 기준 실행
                    try:
                        node.robot_control_base_aligned("1", piece_id, pos, execution_plan, board)
                        print(f"     ✅ Base-aligned execution completed for piece {piece_id}")
                    except Exception as e:
                        print(f"     ❌ Base-aligned execution failed for piece {piece_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                    # 다음 단계 전 잠시 대기
                    time.sleep(1.0)
                    
                except Exception as e:
                    print(f"   ❌ Step {i+1} processing error: {e}")
                    continue

            print(f"\n🎉 Assembly sequence completed after {attempt} planning attempts!")
            print("-" * 80)

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리 작업
        try:
            if 'node' in locals():
                node.destroy_node()
            rclpy.shutdown()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

if __name__ == '__main__':
    main()
