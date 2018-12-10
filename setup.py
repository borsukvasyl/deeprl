from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='deeprl',
    version='0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/borsukvasyl/deeprl',
    license='ISC',
    author='Vasyl Borsuk',
    author_email='vas.borsuk@gmail.com',
    description='Library for Reinforcement Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=[
        "numpy",
        "tensorflow",
        "gym"
    ]
)

# ['deeprl', 'deeprl.utils', 'deeprl.agents', 'deeprl.models', 'deeprl.models.policy',
#               'deeprl.models.qlearning', 'deeprl.models.actorcritic', 'deeprl.policy', 'deeprl.trainers',
#               'deeprl.trainers.policy', 'deeprl.trainers.qlearning', 'deeprl.trainers.actorcritic', 'deeprl.callbacks',
#               'deeprl.environments', 'examples']
