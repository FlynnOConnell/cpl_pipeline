import platform

from setuptools import find_packages, setup

pyreq = ">3.8.0"
if platform.machine() != "arm64":
    pyreq = "3.9.0"

setup(
    name="spk2extract",
    version="0.0.1",
    description="Signal extract utilities.",
    author="Flynn OConnell",
    author_email="Flynnoconnell@gmail.com",
    url="https://www.github.com/Flynnoconnell/spk2extract",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
