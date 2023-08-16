from distutils.core import setup

setup(
    name='verbal_gym',
    version='0.1.0',
    author='Verbal-feedback Team',
    author_email='chinganc@microsoft.com',
    packages=['verbal_gym'],
    url='https://github.com/microsoft/verbal-gym',
    license='MIT LICENSE',
    description='A gym environment for learning with verbal feedback.',
    long_description=open('README.md').read(),
    install_requires=[
        "gym==0.25.2",
        # "Cython==0.29.36",
        "gym-bandits@git+https://github.com/JKCooper2/gym-bandits#egg=gym-bandits",
    ],
    extras_require={
        'gpt': ['openai'],
        'ray': ['ray>=1.5',\
                'numpy<1.24.0' # this is for compatibility with ray (due to pickle)
                ]
    }

)
