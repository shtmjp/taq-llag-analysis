from setuptools import find_packages, setup  # ty: ignore[unresolved-import]


setup(
    name="taq-llag-analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
