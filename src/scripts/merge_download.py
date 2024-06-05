"""

Workaround because I don't have paid Overleaf...



"""

from pathlib import Path
import zipfile
from subprocess import Popen, PIPE
try:
    import showyourwork
except ModuleNotFoundError as e:
    raise RuntimeError('Please install `showyourwork` first. Maybe change the conda environment?') from e

import paths

FROM_OVERLEAF_BRANCH_NAME = 'stage-from-overleaf'
ZIP_FILE_NAME = 'cb-disk-libration-rebound.zip'
DOWNLOADS_PATH = Path.home() / 'Downloads'
ZIP_FILE_PATH = DOWNLOADS_PATH / ZIP_FILE_NAME
SRC_PATH = paths.src
REPO_PATH = paths.root



def get_current_branch_name() -> str:
    return Popen(['git', 'branch', '--show-current'], stdout=PIPE).communicate()[0].decode('utf-8').strip()


if __name__ == '__main__':
    current_branch_name = get_current_branch_name()
    print(f'Current branch name: {get_current_branch_name()}')
    if current_branch_name != FROM_OVERLEAF_BRANCH_NAME:
        raise RuntimeError(f'Current branch name is not {FROM_OVERLEAF_BRANCH_NAME}')
    if not ZIP_FILE_PATH.exists():
        raise RuntimeError(f'Zip file not found at {ZIP_FILE_PATH}')
    print(f'Extracting {ZIP_FILE_PATH} to {SRC_PATH}')
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(SRC_PATH)
    print('Done!')
    print('Deleting zip file')
    ZIP_FILE_PATH.unlink()
    print('Done!')
    print(f'Running showyourwork! from {REPO_PATH}')
    with Popen(['showyourwork', 'build'], cwd=REPO_PATH,stdout=PIPE) as p:
        for line in p.stdout:
            print(line.decode('utf-8').strip())
    print('Now commit the changes and merge it yourself! Check the PDF first and make sure everything worked!')
    
    
    
