import os
import subprocess
import time
import re


def run_command(cmd, retries=3):
    for _ in range(retries):
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            stdout_lines = process.stdout.readlines()
            stderr_lines = process.stderr.readlines()

            for line in stdout_lines:
                print(line, end='')  # Print stdout in real-time
            for line in stderr_lines:
                print(line, end='')  # Print stderr in real-time

            satisfied_lines = [line for line in stdout_lines if "Requirement already satisfied" in line or "Looking in indexes" in line]

            # Check conditions for successful runs
            if process.returncode == 0 or \
                    "Successfully installed" in "".join(stdout_lines) or \
                    "Preparing metadata (setup.py): finished with status 'done'" in "".join(stdout_lines) or \
                    ("Executing transaction: ...working... done" in "".join(stdout_lines) and "# To activate this environment, use" in "".join(stdout_lines)):
                return True
            elif len(satisfied_lines) == len(stdout_lines):  # All lines indicate requirements are already satisfied
                print("All requirements are satisfied. Skipping...")
                return True
            elif "CondaValueError: prefix already exists" in "".join(stderr_lines):
                print("Conda environment already exists. Skipping creation.")
                return True  # If environment already exists, we consider it a successful run for this command
            elif not stderr_lines or stderr_lines.__len__() == 0 or all("CondaValueError: prefix already exists" in line for line in stderr_lines):
                print(f"Command '{cmd}' executed without errors.")
                return True
            elif re.search("Preparing metadata \([a-zA-Z]+?\): finished with status 'done'", "".join(stdout_lines)):
                print(f"Command '{cmd}' executed with status done.")
                return True
            elif "To activate this environment, use" in "".join(stderr_lines):
                print(f"Command '{cmd}' executed with an new env.")
                return True
            else:
                print(f"Command '{cmd}' failed. Retrying...")
                time.sleep(2)  # Adding a delay of 2 seconds between retries

    print(f"Command '{cmd}' failed after {retries} retries. Skipping...")
    return False


def get_conda_env_path(env_name):
    cmd = "conda info --envs"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    env_list, _ = process.communicate()
    env_list = env_list.strip().split('\n')

    # Extracting the path for the specific environment
    for env_line in env_list:
        if env_name in env_line:
            # Splitting by spaces and taking the last item which should be the path
            return env_line.split()[-1]
    return None


if __name__ == '__main__':
    env_name = "medseg"

    commands = [
        f"conda create -n {env_name} python=3.9 -y",
        f"conda activate {env_name} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117",
        f"conda activate {env_name} && pip install git+https://github.com/ChaoningZhang/MobileSAM.git",
        f"conda activate {env_name} && pip install timm",
        f"conda activate {env_name} && pip install git+https://github.com/Kent0n-Li/nnSAM.git"
    ]

    for cmd in commands:
        print(cmd + "####################################################")
        run_command(cmd)
        print("current cmd done")

    # Get the path to the newly created conda environment
    env_path = get_conda_env_path(env_name)
    print(env_path)
    os.chdir(env_path)
    run_command(f"git clone https://github.com/Kent0n-Li/Medical-Image-Segmentation.git")

    run_command(f"conda activate {env_name} && pip install -r {env_path}\\Medical-Image-Segmentation\\requirements.txt")

    print("Script execution completed.")
