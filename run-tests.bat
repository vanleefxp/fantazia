cd "%~dp0/tests"
coverage run -m unittest
coverage html
cd "%~dp0"