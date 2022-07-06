import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name='conv-facenet',
    version='0.0.1',
    author='Ahmed badr',
    author_email='ahmed.k.badr.97@gmail.com',
    description='using convnext as base model for face recognition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ahmedbadr97/conv-facenet',
    license='MIT',
    packages=['convfacenet'],
    package_dir={
        'convfacenet': 'src/convfacenet'},
    install_requires=['torch','torchvision','numpy','opencv-python','pandas','gdown','Pillow'],
)