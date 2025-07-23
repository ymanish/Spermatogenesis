from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="src.cyt_script.tricodec",          
        sources=["src/cyt_script/tricodec.pyx"],
        language="c",
    ),
    Extension(
        name="src.cyt_script.edit_tricodec",
        sources=["src/cyt_script/edit_tricodec.pyx"],
        language="c",
    ),
]


setup(
        name="spermatogenesis",
        version='0.1',
        packages=find_packages(),
        author='Manish Yadav',
        author_email='manish20072013 at gmail dot com',
        description='Spermatogenesis simulation package: Nucleosome replacement and protamine binding dynamics',
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        license='GPL3', 
        python_requires=">=3.12",
        setup_requires=["Cython>=0.29"],
        ext_modules=cythonize(extensions, language_level=3, annotate=True)

    )