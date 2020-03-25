from setuptools import setup, find_packages

setup(
    name='planetpy',
    version='1.0',
    description='Deep Telemetry Planet Image Download Script',
    author='Eric Magliarditi',
    author_email='ericmags@mit.edu',
    packages=['planetpy'],
    install_requires=['numpy','pandas'],
    scripts=['bin/download_images.py']
)

