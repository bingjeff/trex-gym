from setuptools import setup

setup(
    name='stan_trex',
    version='',
    packages=['trex_gym'],
    package_dir={'': 'trex-gym'},
    url='',
    license='MIT',
    author='Jeffrey T. Bingham',
    author_email='bingjeff@gmail.com',
    description='', install_requires=['numpy', 'tensorflow', 'absl-py', 'baselines', 'pybullet', 'gym', 'joblib', 'PIL',
                                      'absl']
)
