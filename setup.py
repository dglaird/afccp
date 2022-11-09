from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

install_requires = ['sdv==0.14.0', 'Pyomo==6.4.0', 'python-pptx==0.6.21', 'packaging==21.3', 'openpyxl==3.0.9',
                    'xlsxwriter==3.0.3', 'sklearn==0.0']

setup(name='afccp',
      version='0.1',
      description='Air Force Cadet Career Problem',
      author='Griffen Laird',
      author_email='griffenlaird007@gmail.com',
      url='https://github.com/dglaird/afccp',
      install_package_date=True,
      install_requires=install_requires,
      license='MIT license',
      keywords='afccp AFCCP',
      packages=find_packages(include=['afccp', 'afccp.*']),
      python_requires='>=3.6,<3.10'
     )