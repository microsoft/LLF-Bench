from distutils.core import setup

setup(
    name='llfbench',
    version='0.1.0',
    author='LLF-Bench Team',
    author_email='chinganc@microsoft.com',
    packages=['llfbench'],
    url='https://github.com/microsoft/LLF-Bench',
    license='MIT LICENSE',
    description='A gym environment for learning with verbal feedback.',
    long_description=open('README.md').read(),
    install_requires=[
        "tqdm",
        "gym>=0.25.2,<0.26.0",
        "gymnasium==0.29.1",
        "parse==1.19.1",
        # "Cython==0.29.36",
        "gym-bandits@git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits",
        # poem
        "cmudict",
        "syllables",
        # loss_landscape
        "jax",
        "jaxlib",
        # highway
        "highway-env",
        # movie
        'requests==2.31.0'
    ],
    extras_require={
        'metaworld': ['metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld'],
        'alfworld': [ 'fast-downward@https://github.com/MarcCote/downward/archive/faster_replan.zip',
                      'textworld@https://github.com/MarcCote/TextWorld/archive/handcoded_expert_integration.zip',
                      'alfworld@git+https://github.com/chinganc/alfworld.git@master#egg=alfworld',
                      'werkzeug==2.2.2',
                      'flask==2.0.2',
                      'click==7.1.2',
        ]
    }
)
