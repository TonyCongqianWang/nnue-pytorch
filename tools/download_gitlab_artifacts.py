import requests
import argparse
import os
import re
import zipfile
import sys

def get_default_project():
    """Attempts to extract the GitLab project path from local .git/config."""
    try:
        if os.path.exists(".git/config"):
            with open(".git/config", "r") as f:
                content = f.read()
                match = re.search(r"url = .*gitlab\.com[:/](.*?)(?:\.git)?\n", content)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None

def download_and_extract(project_path, pipeline_id, output_dir, unzip=True, token=None):
    encoded_path = project_path.replace("/", "%2F")
    api_url = f"https://gitlab.com/api/v4/projects/{encoded_path}/pipelines/{pipeline_id}/jobs"

    headers = {"PRIVATE-TOKEN": token} if token else {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--> Target: {project_path} | Pipeline: {pipeline_id}")

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        jobs = response.json()
    except Exception as e:
        print(f"Error fetching job list: {e}")
        return

    for job in jobs:
        if job.get('artifacts_file'):
            job_id = job['id']
            # Clean filename/foldername
            clean_name = re.sub(r'[^\w\-_\.]', '_', job['name'])
            zip_path = os.path.join(output_dir, f"{clean_name}_{job_id}.zip")
            extract_path = os.path.join(output_dir, clean_name)

            download_url = f"https://gitlab.com/api/v4/projects/{encoded_path}/jobs/{job_id}/artifacts"

            print(f"Downloading: {job['name']}...")
            with requests.get(download_url, headers=headers, stream=True) as r:
                if r.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if unzip:
                        print(f"  Unzipping to: {extract_path}...")
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_path)
                            os.remove(zip_path) # Clean up zip after extraction
                        except zipfile.BadZipFile:
                            print(f"  Error: {zip_path} is not a valid zip file.")
                else:
                    print(f"  Failed: HTTP {r.status_code}")

if __name__ == "__main__":
    default_proj = get_default_project()

    parser = argparse.ArgumentParser(description="Download and unzip GitLab artifacts.")
    parser.add_argument("pipeline", help="The Pipeline ID")
    parser.add_argument("-p", "--project", default=default_proj,
                        help=f"Project path (Default: {default_proj})")
    parser.add_argument("-o", "--output", default="artifacts",
                        help="Output directory (default: 'artifacts')")
    parser.add_argument("--no-unzip", action="store_false", dest="unzip",
                        help="Disable automatic unzipping")
    parser.set_defaults(unzip=True)
    parser.add_argument("--token", help="GitLab Personal Access Token")

    args = parser.parse_args()

    if not args.project:
        print("Error: Could not detect project. Use -p <path>")
        sys.exit(1)

    download_and_extract(args.project, args.pipeline, args.output, args.unzip, args.token)