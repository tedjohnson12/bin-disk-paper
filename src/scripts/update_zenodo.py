"""
Write the current database to a zenodo record

Much of this is taken from https://github.com/showyourwork/showyourwork/src/showyourwork/zenodo.py

"""
import datetime
import requests
import subprocess
import json
import yaml
import os

import paths

DB_PATH = paths.data / 'mc.db'
SYW_PATH = paths.root / 'showyourwork.yml'
SANDBOX_TOKEN = os.environ['SANDBOX_TOKEN']

def get_latest_id() -> str:
    with open(SYW_PATH, 'r', encoding='utf-8') as f:
        yml = yaml.safe_load(f)
        datasets = yml['datasets']
        assert len(datasets) == 1
        key:str = list(datasets.keys())[0]
        latest_id = key.split('zenodo.')[-1]
        return latest_id


if __name__ == '__main__':
    latest_id = get_latest_id()
    params = {'access_token': SANDBOX_TOKEN}
    r = requests.get(f'https://sandbox.zenodo.org/api/deposit/depositions/{latest_id}',
                      params=params,
                      json={},
                      timeout=20)
    newversion_url = r.json()['links']['newversion']
    metadata=r.json()['metadata']
    metadata['publication_date'] = datetime.date.today().strftime("%Y-%m-%d")
    # create a new version
    draft = requests.post(newversion_url,
                      params=params,
                      data=json.dumps({'metadata': metadata}),
                      json={},
                      timeout=20).json()
    bucket_url = draft['links']['bucket']
    concept_id = draft['conceptrecid']
    file_to_upload = str(DB_PATH)
    res = subprocess.run(
                [
                    "curl",
                    "-f",
                    '--progress-bar',
                    "-o",
                    "/dev/null",
                    "--upload-file",
                    str(file_to_upload),
                    "--request",
                    "PUT",
                    f"{bucket_url}/mc.db?access_token={SANDBOX_TOKEN}",
                ],
                check=True
            )
    r = requests.get(
            "https://sandbox.zenodo.org/api/deposit/depositions",
            params={
                "q": f"conceptrecid:{concept_id}",
                "all_versions": 1,
                "access_token": SANDBOX_TOKEN,
            },
            timeout=20
        )
    for data in r.json():
        if not data["submitted"]:
            break
    else:
        raise RuntimeError('Not submitted')
    version_id = data["id"]
    # r = requests.post(
    #     f"https://sandbox.zenodo.org/api/deposit/depositions/{version_id}/actions/publish",
    #     params=params,
    #     data={'metadata': metadata},
    #     timeout=20
    # )
    # r.raise_for_status()
    with open(SYW_PATH, 'r', encoding='utf-8') as fold:
        yml_old = yaml.safe_load(fold)
        datasets = yml_old['datasets']
        assert len(datasets) == 1
        key:str = list(datasets.keys())[0]
        latest_id = key.split('zenodo.')[-1]
        value = datasets[key]
    with open(SYW_PATH, 'w', encoding='utf-8') as fout:
        yml_old['datasets'] = {key.replace(latest_id, str(version_id)): value}
        yaml.dump(yml_old, fout)