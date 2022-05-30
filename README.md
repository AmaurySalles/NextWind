# Streamlit
https://share.streamlit.io/nicolashenriquez/projectwind/app_main.py


# Install

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:AmaurySalles/nextwind.git
cd nextwind
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
nextwind-run
```
