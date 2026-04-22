from setuptools import find_packages, setup

package_name = 'gwtd'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaehun',
    maintainer_email='jaehunchoe827@gmail.com',
    description='guidewire tip detection.',
    license='TODO: License declaration',
    scripts=[],
    entry_points={
        'console_scripts': [
        ],
    },
)
