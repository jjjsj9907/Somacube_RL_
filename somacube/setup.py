from setuptools import find_packages, setup
import glob
import os
package_name = 'somacube'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob.glob('resource/*')),
        # ('share/' + package_name + '/launch', glob.glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jsj2204',
    maintainer_email='jjjsj9907@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_move = somacube.robot_move:main',
            'detection = somacube.detection:main',
            'RL_with_robot = somacube.RL_with_robot:main',
            'RL_with_robot2 = somacube.RL_with_robot2:main',
            'RL_with_robot3 = somacube.RL_with_robot3:main',
            'secret = somacube.re_game_somacube:main',
            'secret2 = somacube.re_game_somacube_three_execute:main',
            'sim = somacube.move_simul:main',
            'take1 = somacube.somacube_take_1:main',
            'take2 = somacube.somacube_take_2:main',
            'ultra = somacube.ultra_RL:main',
        ],
    },
)
