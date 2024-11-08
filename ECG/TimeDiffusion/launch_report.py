from ai4ha.util import load_jobs
import subprocess
from glob import glob
import datetime
import os
# import argparse

LOCAL_PATH = "/home/bejar/PycharmProjects/misiones/Series/Models/"
MODELS_PATH = "/home/bejar/PycharmProjects/misiones/Series/Models/"

if __name__ == "__main__":
    model = 'TimeDiffusion'
    jobfiles = load_jobs(f"{MODELS_PATH}/{model}/jobs/results.jobs")

    for job in jobfiles:
        print(f"Processing {job}")
        subprocess.run(f"python {MODELS_PATH}/{model}/SaveResults.py {job}",
                       shell=True)
        print(f"Processed {job}")

    dirs = glob(f"{MODELS_PATH}/{model}/0results/*")
    for d in dirs:
        drep = glob(f"{d}/*")
        frep = open(f"{d}/report.md", "w")
        frep.write(f"# Dataset = {d.split('/')[-1]}\n\n")
        for dr in sorted(drep):
            m_time = os.path.getmtime(dr)
            frep.write('---\n\n')
            frep.write(f"##  {str(datetime.datetime.fromtimestamp(m_time))[:10]} - {dr.split('/')[-1]} \n\n")
            img = glob(f"{dr}/*")
            for i in sorted(img):
                if i.split('.')[-1] == 'png':
                    frep.write(
                        f"![{i.split('/')[-1]}](./{dr.split('/')[-1]}/{i.split('/')[-1]})\n\n"
                    )
