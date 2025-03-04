from setuptools import setup, find_packages

setup(
    name="InklessAI",
    version="0.1.0",
    author="Somesh Panchal",
    author_email="somesh.panchal00@example.com",
    description="AI-powered multimodal system for handwriting recognition, face analysis, and NLP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "mediapipe",
        "tensorflow",
        "torch",
        "transformers",
        "sympy",
        "fastapi",
        "uvicorn",
        "streamlit",
        "onnxruntime"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
