from setuptools import setup, find_packages

setup(
    name="tutils",
    version="0.1.0",
    description="常用机器学习 & 数据科学工具包",
    author="Your Name",
    author_email="your@email.com",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "colorlog",
        "tqdm",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)