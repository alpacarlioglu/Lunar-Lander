from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'

def get_packages(requirements_path):
    requirements = []
    with open(requirements_path, 'r') as file:
        requirements = file.readlines()
        requirements = [x.replace('\n', '') for x in requirements] # Replace spaces with empty.
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='Lunar Lander',
    author_email='alpacarlioglu@gmail.com',
    version='1.0.0',
    author="Alp Acarlioglu",
    install_requires= get_packages('requirements.txt')  # Or whatever your file path is
)