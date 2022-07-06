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
    include_package_data=True,
    url='https://github.com/ahmedbadr97/conv-facenet',
    license='MIT',
    packages=['convfacenet','convfacenet.face_descriptor','convfacenet.face_detector'],
    package_dir={
        'convfacenet': 'src/convfacenet','convfacenet.face_descriptor': 'src/convfacenet/face_descriptor','convfacenet.face_detector':'src/convfacenet/face_detector'},
    install_requires=['torch','torchvision','numpy','opencv-python','pandas','gdown','Pillow'],
)