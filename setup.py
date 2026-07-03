from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clip-medsam",
    version="0.1.0",
    author="Raihan Uddin Ahmed",
    author_email="raihan.uddin@example.com",
    description="Vision Language Model for Medical Image Segmentation using CLIP and SAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raihan-uddinahmed/CLIP-MedSAM-trial",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "clip-interrogator>=0.5.0",
        "open-clip-torch>=2.20.0",
        "nibabel>=3.0.0",
        "pydicom>=2.3.0",
        "SimpleITK>=2.1.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
)
