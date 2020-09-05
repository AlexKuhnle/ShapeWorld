from setuptools import setup


setup(
    name='ShapeWorld',
    version='0.1',
    url='https://github.com/AlexKuhnle/ShapeWorld',
    download_url='',
    author='Alexander Kuhnle',
    author_email='alexkuhnle@t-online.de',
    license='MIT',
    description='Configurable abstract data generation with a focus on visually grounded language data for deep learning evaluation.',
    long_description='Configurable abstract data generation with a focus on visually grounded language data for deep learning evaluation.',
    keywords=[],
    platforms=['linux', 'mac'],
    packages=['shapeworld'],
    install_requires=['numpy', 'pillow', 'six'],
    extras_require={
        'full': ['tensorflow', 'wget'],
        'full-gpu': ['tensorflow-gpu', 'wget']
    }
)
