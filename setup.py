from distutils.core import setup

setup(
    name='raanova',
    version='0.1',
    description='Core',
    author='Aurélien Bück-Kaeffer, Mila Pourali, Andrew Rambidis',
    author_email='',
    package_dir={'': 'src'},
    packages=['raanova'],
    extras_require={
        'dev': [
            'unittest'
        ]
    }
)
