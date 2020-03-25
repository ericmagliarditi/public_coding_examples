from setuptools import setup, find_packages

setup(
    name='landpy',
    version='1.0',
    description='Deep Telemetry & ML Project',
    author='Eric Magliarditi',
    author_email='ericmags@mit.edu',
    packages=['landpy'],
    install_requires=[
    	'numpy','pandas', 'torch', 'tqdm',
    	'argparse', 'scikit-learn'
    ],
    scripts=['bin/train.py',
    'bin/evaluate.py',
    'app/land_use_dashboard.py',
    'app/simple_land_use_dashboard.py']
)

