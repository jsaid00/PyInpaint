from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'tqdm',
    'scipy',
    'SimpleITK',
    'matplotlib',
    'Pillow',
]

setup(
    name="PyInpaint",
    version="1.2.0",
    description="An enhanced lightweight image inpainting tool",
    url="https://github.com/jsaid00/PyInpaint",
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    author="Jawher Said",
    author_email="said@zib.de",
    license="MIT",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pyinpaint=pyinpaint.__main__:main",
        ]
    },
)
