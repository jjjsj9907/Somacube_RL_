#!/usr/bin/env python3
"""
DSR_ROBOT2 기반 좌표계 테스트 코드
"""

import rclpy
import DR_init
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

def apply_rotation_manually(start_pose, zyz_delta):
    """기존 회전 적용 함수 (scipy 사용)"""
    try:
        start_euler = start_pose[3:]
        r_start = R.from_euler('ZYZ', start_euler, degrees=True)
        r_delta = R.from_euler('ZYZ', zyz_delta, degrees=True)
        r_final = r_start * r_delta
        final_euler = r_final.as_euler('zyz', degrees=True)
        final_euler = [(angle + 180) % 360 - 180 for angle in final_euler]
        final_pose = start_pose[:3] + list(final_euler)
        return final_pose
    except Exception as e:
        print(f"Rotation application error: {e}")
        return start_pose

def apply_rotation_y_inverted(start_pose, zyz_delta):
    """Y축 반전 회전 함수"""
    try:
        z1, y, z2 = zyz_delta
        y = -y  # Y축 방향 반전
        corrected_delta = [z1, y, z2]
        
        start_euler = start_pose[3:]
        r_start = R.from_euler('ZYZ', start_euler, degrees=True)
        r_delta = R.from_euler('ZYZ', corrected_delta, degrees=True)
        r_final = r_start * r_delta
        final_euler = r_final.as_euler('zyz', degrees=True)
        final_euler = [(angle + 180) % 360 - 180 for angle in final_euler]
        final_pose = start_pose[:3] + list(final_euler)
        return final_pose
    except Exception as e:
        print(f"Y-inverted rotation error: {e}")
        return start_pose

def apply_rotation_direct(start_pose, zyz_delta):
    """직접 각도 덧셈 방식"""
    try:
        z1, y, z2 = zyz_delta
        current_z1, current_y, current_z2 = start_pose[3:]
        
        final_z1 = current_z1 + z1
        final_y = current_y + y
        final_z2 = current_z2 + z2
        
        # 각도 정규화 (-180 ~ 180)
        final_z1 = ((final_z1 + 180) % 360) - 180
        final_y = ((final_y + 180) % 360) - 180  
        final_z2 = ((final_z2 + 180) % 360) - 180
        
        final_pose = start_pose[:3] + [final_z1, final_y, final_z2]
        return final_pose
    except Exception as e:
        print(f"Direct rotation error: {e}")
        return start_pose

def test_single_axis_rotations(movej, movel, get_current_posx, posx):
    """각 축별 90도 회전 테스트"""
    
    print("\n" + "="*60)
    print("각 축별 90도 회전 테스트")
    print("="*60)
    
    # 안전한 테스트 위치
    test_base = [450, 0, 300, 0, 180, 0]
    print(f"테스트 기준 위치로 이동: {test_base}")
    movel(posx(test_base), vel=30, acc=30)
    time.sleep(1.0)
    
    test_rotations = [
        ([90, 0, 0], "Z1축 +90도"),
        ([0, 90, 0], "Y축 +90도"), 
        ([0, 0, 90], "Z2축 +90도"),
        ([-90, 0, 0], "Z1축 -90도"),
        ([0, -90, 0], "Y축 -90도"),
        ([0, 0, -90], "Z2축 -90도")
    ]
    
    for delta, description in test_rotations:
        print(f"\n--- {description} ---")
        
        # 방법 1: 기존 scipy 방법
        target_scipy = apply_rotation_manually(test_base, delta)
        print(f"Scipy 계산: {[round(x,1) for x in target_scipy[3:]]}")
        
        # 방법 2: 직접 덧셈
        target_direct = apply_rotation_direct(test_base, delta)
        print(f"직접 덧셈: {[round(x,1) for x in target_direct[3:]]}")
        
        # 실제 이동 (scipy 방법)
        try:
            print("이동 중...")
            movel(posx(target_scipy), vel=20, acc=20)
            time.sleep(1.0)
            
            actual_pose = get_current_posx()[0]
            print(f"실제 결과: {[round(x,1) for x in actual_pose[3:]]}")
            
            # 오차 계산
            angle_diff = [abs(target_scipy[i+3] - actual_pose[i+3]) for i in range(3)]
            print(f"각도 오차: {[round(x,1) for x in angle_diff]}")
            
        except Exception as e:
            print(f"이동 실패: {e}")
        
        # 기준 자세로 복귀
        print("기준 위치로 복귀...")
        movel(posx(test_base), vel=30, acc=30)
        time.sleep(0.5)
        
        input("다음 테스트를 위해 Enter를 누르세요...")

def test_b_piece_rotation(movej, movel, get_current_posx, posx):
    """B 조각 ZYZ(-180°, 180°, 0°) 회전 테스트"""
    
    print("\n" + "="*60)
    print("B 조각 회전 테스트: ZYZ(-180°, 180°, 0°)")
    print("="*60)
    
    # 안전한 테스트 위치
    test_base = [450, 0, 300, 0, 180, 0]
    print(f"테스트 기준 위치: {test_base}")
    movel(posx(test_base), vel=30, acc=30)
    time.sleep(1.0)
    
    rotation_target = [-180, 180, 0]
    
    # 1. 시뮬레이션 예상 결과
    print("\n1. 시뮬레이션 예상 결과:")
    b_piece = np.array([[0,0,0], [1,0,0], [1,1,0], [1,1,1]])
    r = R.from_euler('ZYZ', rotation_target, degrees=True)
    rotated_piece = np.array([r.apply(coord) for coord in b_piece])
    normalized = rotated_piece - rotated_piece.min(axis=0)
    normalized_int = np.round(normalized).astype(int)
    print(f"회전된 B 조각 좌표:\n{normalized_int}")
    
    # 2. 세 가지 방법으로 테스트
    methods = [
        ("원본 scipy", apply_rotation_manually),
        ("Y축 반전", apply_rotation_y_inverted),
        ("직접 덧셈", apply_rotation_direct)
    ]
    
    for method_name, rotation_func in methods:
        print(f"\n--- {method_name} 방법 ---")
        
        target = rotation_func(test_base, rotation_target)
        print(f"계산 결과: {[round(x,1) for x in target[3:]]}")
        
        try:
            movel(posx(target), vel=20, acc=20)
            time.sleep(1.0)
            
            actual = get_current_posx()[0]
            print(f"실제 결과: {[round(x,1) for x in actual[3:]]}")
            
            # B 조각 좌표 계산 (해당 방법으로)
            if method_name == "Y축 반전":
                corrected_target = [-180, -180, 0]
            else:
                corrected_target = rotation_target
                
            r_test = R.from_euler('ZYZ', corrected_target, degrees=True)
            rotated_test = np.array([r_test.apply(coord) for coord in b_piece])
            normalized_test = rotated_test - rotated_test.min(axis=0)
            normalized_test_int = np.round(normalized_test).astype(int)
            print(f"B 조각 예상: {normalized_test_int.tolist()}")
            
        except Exception as e:
            print(f"{method_name} 이동 실패: {e}")
        
        # 기준으로 복귀
        movel(posx(test_base), vel=30, acc=30)
        time.sleep(1.0)
        
        input(f"{method_name} 테스트 완료. 다음으로 계속하려면 Enter...")

def test_step_by_step_rotation(movej, movel, get_current_posx, posx):
    """단계별 회전 테스트"""
    
    print("\n" + "="*60)
    print("단계별 회전 테스트")
    print("="*60)
    
    test_base = [450, 0, 300, 0, 180, 0]
    movel(posx(test_base), vel=30, acc=30)
    time.sleep(1.0)
    
    print("1단계: Z1 = -180°")
    step1 = apply_rotation_manually(test_base, [-180, 0, 0])
    print(f"목표: {[round(x,1) for x in step1[3:]]}")
    movel(posx(step1), vel=20, acc=20)
    time.sleep(1.0)
    
    current_pose = get_current_posx()[0]
    print(f"실제: {[round(x,1) for x in current_pose[3:]]}")
    
    input("2단계로 계속하려면 Enter...")
    
    print("2단계: Y = +180°")
    step2 = apply_rotation_manually(current_pose, [0, 180, 0])
    print(f"목표: {[round(x,1) for x in step2[3:]]}")
    movel(posx(step2), vel=20, acc=20)
    time.sleep(1.0)
    
    final_pose = get_current_posx()[0]
    print(f"최종 결과: {[round(x,1) for x in final_pose[3:]]}")

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("coordinate_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import (
            set_tool,
            set_tcp,
            movej,
            movel,
            get_current_posx,
        )
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return
    
    # 로봇 초기화
    JReady = [0, 0, 90, 0, 90, 0]
    set_tool("Tool Weight_2FG")
    set_tcp("2FG_TCP")
    
    print("로봇 좌표계 테스트를 시작합니다.")
    print("안전을 위해 로봇 주변을 확인하세요.")
    input("계속하려면 Enter를 누르세요...")
    
    # HOME으로 이동
    print("HOME 위치로 이동...")
    movej(JReady, vel=VELOCITY, acc=ACC)
    time.sleep(1.0)
    
    while rclpy.ok():
        print("\n" + "="*50)
        print("테스트 메뉴:")
        print("1. 각 축별 90도 회전 테스트")
        print("2. B 조각 회전 테스트 (3가지 방법)")
        print("3. 단계별 회전 테스트")
        print("4. 기본 움직임 테스트")
        print("5. HOME으로 복귀")
        print("q. 종료")
        print("="*50)
        
        choice = input("선택하세요: ").strip()
        
        try:
            if choice == "1":
                test_single_axis_rotations(movej, movel, get_current_posx, posx)
            elif choice == "2":
                test_b_piece_rotation(movej, movel, get_current_posx, posx)
            elif choice == "3":
                test_step_by_step_rotation(movej, movel, get_current_posx, posx)
            elif choice == "4":
                # 기본 움직임 테스트
                print("기본 pick & place 테스트...")
                pos1 = posx([496.06, 93.46, 296.92, 20.75, 179.00, 19.09])
                pos2 = posx([548.70, -193.46, 96.92, 20.75, 179.00, 19.09])
                
                movej(JReady, vel=VELOCITY, acc=ACC)
                movel(pos1, vel=VELOCITY, acc=ACC)
                movel(pos2, vel=VELOCITY, acc=ACC)
                movej(JReady, vel=VELOCITY, acc=ACC)
                print("기본 테스트 완료!")
            elif choice == "5":
                print("HOME으로 복귀...")
                movej(JReady, vel=VELOCITY, acc=ACC)
            elif choice.lower() == "q":
                print("테스트를 종료합니다.")
                break
            else:
                print("잘못된 선택입니다.")
                
        except KeyboardInterrupt:
            print("\n사용자가 테스트를 중단했습니다.")
            break
        except Exception as e:
            print(f"테스트 오류: {e}")
    
    # 종료 시 HOME으로 복귀
    try:
        print("HOME 위치로 복귀...")
        movej(JReady, vel=VELOCITY, acc=ACC)
    except:
        pass
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()